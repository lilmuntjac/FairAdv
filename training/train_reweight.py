import time
from pathlib import Path

import torch

from utils.metrics_utils import (
    normalize, get_confusion_matrix_counts, get_rights_and_wrongs_counts, 
    print_binary_model_summary, print_multiclass_model_summary
)

class ReWeightTrainer:
    def __init__(self, config, train_loader, val_loader,
                 model, criterion, optimizer, scheduler, save_path, device='cpu'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.model_type = config['dataset']['type']
        self.task_name = config['dataset'].get('task_name', 'unspecified')
        self.criterion = criterion
        self.optimizer = optimizer 
        self.scheduler = scheduler # unused
        self.save_path = Path(save_path) # Ensure it's a Path object
        self.device = device
        # reweight (Label Bias Correcting)
        self.num_epochs = config['training'].get('final_epoch', 5)
        self.iteration = config['reweight'].get('iteration', 10)
        self.eta = config['reweight'].get('eta', 0.001)
        protected_attr = config['dataset']['protected_attr']
        if isinstance(protected_attr, str):
            num_protected_attributes = 1
        elif isinstance(protected_attr, list):
            num_protected_attributes = len(protected_attr)
        else:
            raise ValueError("The 'protected_attr' should be either a string or a list.")
        num_target_attributes = config['dataset'].get('num_outputs', 1)
        self.multipliers = torch.zeros((2 * num_protected_attributes, 2 * num_target_attributes), 
                                       device=self.device)
        self.weights = self.initialize_weights(config)  # Initialize weights as None

    def initialize_weights(self, config):
        num_batches = len(self.train_loader)
        batch_size = config['training']['batch_size']
        weights = torch.ones((num_batches, batch_size), device=self.device)
        return weights
    
    # Only support binary prodictions !!
    def debias_weights(self, labels):
        protected = labels[:, -1]  # in shape (N,) N here is the entire dataset
        original_labels = labels[:, :-1].squeeze(1)  # in shape (N,)

        exponents = torch.zeros(len(original_labels), device=self.device)
        num_groups = self.multipliers.size(0) // 2
        
        for i in range(num_groups):
            tpr_exponent = self.multipliers[2 * i, 0]  # TPR violation multiplier for group i
            fpr_exponent = self.multipliers[2 * i, 1]  # FPR violation multiplier for group i
    
            # For positive labels, apply TPR multiplier
            tpr_mask = (protected == i) & (original_labels == 1)
            exponents[tpr_mask] -= tpr_exponent
    
            # For negative labels, apply FPR multiplier
            fpr_mask = (protected == i) & (original_labels == 0)
            exponents[fpr_mask] -= fpr_exponent

        # Update the weights using the sigmoid of the exponents
        self.weights = torch.sigmoid(exponents)

    def compute_violations(self, predictions, labels):
        protected = labels[:, -1]  # Shape (N,)
        original_labels = labels[:, :-1].squeeze(1)  # Shape (N,)

        num_groups  = self.multipliers.size(0)
        violations = torch.zeros_like(self.multipliers, device=self.device)

        # Calculate overall TPR and FPR
        overall_tp, overall_fp = 0, 0
        overall_tn, overall_fn = 0, 0

        for i in range(len(predictions)):
            if predictions[i] == 1 and original_labels[i] == 1:
                overall_tp += 1
            if predictions[i] == 1 and original_labels[i] == 0:
                overall_fp += 1
            if predictions[i] == 0 and original_labels[i] == 0:
                overall_tn += 1
            if predictions[i] == 0 and original_labels[i] == 1:
                overall_fn += 1
        overall_tpr = overall_tp / (overall_tp + overall_fn + 1e-6)
        overall_fpr = overall_fp / (overall_fp + overall_tn + 1e-6)

        for g in range(num_groups):
            group_tp, group_fp = 0, 0
            group_tn, group_fn = 0, 0
            # Calculate TPR and FPR for the protected group
            for i in range(len(predictions)):
                if protected[i] == g:
                    if predictions[i] == 1 and original_labels[i] == 1:
                        group_tp += 1
                    if predictions[i] == 1 and original_labels[i] == 0:
                        group_fp += 1
                    if predictions[i] == 0 and original_labels[i] == 0:
                        group_tn += 1
                    if predictions[i] == 0 and original_labels[i] == 1:
                        group_fn += 1
            group_tpr = group_tp / (group_tp + group_fn + 1e-6)
            group_fpr = group_fp / (group_fp + group_tn + 1e-6)

            # Calculate TPR and FPR violations
            tpr_violation = group_tpr - overall_tpr # violate if higher than group
            fpr_violation = -(group_fpr - overall_fpr) # violate if lower than group
            violations[g, 0] = tpr_violation
            violations[g, 1] = fpr_violation

        return violations

    def compute_loss(self, outputs, labels):
        # To multiply by the weight, reduction should already be 'none'
        if self.model_type == 'binary':
            loss = self.criterion(outputs, labels[:, :-1].float())
        elif self.model_type == 'multi-class':
            loss = self.criterion(outputs, labels[:, :-1].squeeze(1))
        else:
            raise ValueError(f"Invalid dataset type: {self.model_type}")
        return loss
    
    def compute_stats(self, outputs, labels):
        if self.model_type == 'binary':
            # The binary model does not include a sigmoid function in the output. 
            # Remember to pass the output through the sigmoid function first, 
            # then obtain the predictions from it.
            outputs = torch.sigmoid(outputs)
            predicted = (outputs > 0.5).int()
            stats = get_confusion_matrix_counts(predicted, labels)
        elif self.model_type == 'multi-class':
            _, predicted = torch.max(outputs, 1)
            stats = get_rights_and_wrongs_counts(predicted, labels)
        else:
            raise ValueError(f"Invalid dataset type: {self.model_type}")
        return stats

    def show_stats(self, train_stats, val_stats):
        if self.model_type == 'binary':
            print_binary_model_summary(train_stats, val_stats, self.task_name)
        elif self.model_type == 'multi-class':
            print_multiclass_model_summary(train_stats, val_stats, self.task_name)
        else:
            raise ValueError(f"Invalid dataset type: {self.model_type}")
    
    # Train the model for one epoch.
    def train_epoch(self, last=False):
        self.model.train()
        total_loss, total_stats = 0, None
        all_predictions, all_labels = [], []

        for batch_idx, (images, labels, *_) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            images = normalize(images)
            batch_weights = self.weights[batch_idx].to(self.device)  # Select weights for the current batch
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = batch_weights * self.compute_loss(outputs, labels)
            loss = loss.mean()  # Ensure the loss is a scalar
            loss.backward()
            self.optimizer.step()

            if last:
                # The binary model does not include a sigmoid function in the output. 
                # Remember to pass the output through the sigmoid function first, 
                # then obtain the predictions from it.
                predicted = (torch.sigmoid(outputs) > 0.5).int()
                all_predictions.append(predicted.detach())
                all_labels.append(labels.detach())
                stats = self.compute_stats(outputs, labels)
                # update the total loss and total stats for this epoch
                total_loss += loss.item()
                total_stats = stats if total_stats is None else total_stats + stats
        if last:
            avg_loss = total_loss / (batch_idx + 1)
            all_predictions = torch.cat(all_predictions)
            all_labels = torch.cat(all_labels)
            return avg_loss, total_stats, all_predictions, all_labels
        else:
            return None

    # Validate the model for one epoch.
    def validate_epoch(self):
        self.model.eval()
        total_loss, total_stats = 0, None
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                images = normalize(images)
                outputs = self.model(images)
                loss = self.compute_loss(outputs, labels).mean() # Reduce
                stats = self.compute_stats(outputs, labels)
                # update the total loss and total stats for this epoch
                total_loss += loss.item()
                total_stats = stats if total_stats is None else total_stats + stats
            avg_loss = total_loss / (batch_idx + 1)
        return avg_loss, total_stats


    def run(self, start_epoch, final_epoch, total_train_stats, total_val_stats):
        # This trainer has iterations outside the epoch and only saves per iteration. 
        # Be careful not to confuse it with other trainers.
        # The start_epoch and final_epoch argument would not work
        total_start_time = time.perf_counter()

        for iteration in range(self.iteration):
            print(f"Iteration {iteration + 1}/{self.iteration}")
            iter_start_time = time.perf_counter()
            # Train for an iteration
            for epoch in range(self.num_epochs):
                if epoch == self.num_epochs - 1:  # Last epoch of the iteration
                    train_loss, train_stats, all_predictions, all_labels = self.train_epoch(last=True)
                    val_loss, val_stats = self.validate_epoch()
                    # Use all_predictions and all_labels for updating weights and computing violations
                else:
                    self.train_epoch()
            # Update weights and multipliers
            violations = self.compute_violations(all_predictions, all_labels)
            print(f"Violations: {violations}")
            print(f"Multipliers before updating: {self.multipliers}")
            self.multipliers += self.eta * violations
            print(f"Multipliers after updating: {self.multipliers}")

            self.debias_weights(all_labels)
            print(f"Weight stats: min={self.weights.min().item()}, max={self.weights.max().item()}, mean={self.weights.mean().item()}")
            print(f"Weight distribution: {torch.histc(self.weights, bins=10, min=0, max=1)}")


            total_iter_time = time.perf_counter() - iter_start_time
            print(f"Total Time: {total_iter_time:.2f} seconds ({total_iter_time / 60:.2f} minutes)")
            print(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            self.show_stats(train_stats, val_stats)
            print('-'*120)
            # Concatenate the new iteration's stats along the iteration dimension
            total_train_stats = torch.cat((total_train_stats, train_stats.unsqueeze(0)), dim=0)
            total_val_stats   = torch.cat((total_val_stats,   val_stats.unsqueeze(0)), dim=0)
            # Save model checkpoint per epoch
            checkpoint_path = self.save_path / f'checkpoint_epoch_{iteration + 1:04d}.pth'
            torch.save({
                'epoch': iteration + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
            }, checkpoint_path)

        # Save the training stats
        stats_path = self.save_path / f'stats_end_{iteration + 1:04d}.pt'
        torch.save({
            'epoch': iteration + 1,
            'train': total_train_stats,
            'val'  : total_val_stats,
        }, stats_path)
        total_time = time.perf_counter() - total_start_time
        print(f"Total Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
