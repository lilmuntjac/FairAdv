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
        self.multipliers = torch.zeros(2 * num_protected_attributes, device=self.device)
        self.weights = self.initialize_weights(config)  # Initialize weights as None

    def initialize_weights(self, config):
        num_batches = len(self.train_loader)
        batch_size = config['training']['batch_size']
        weights = torch.ones((num_batches, batch_size), device=self.device)
        return weights
    
    # Only support binary prodictions !!
    def debias_weights(self, labels):
        protected = labels[:, -1] # in shape (N,) N here is the entire dataset
        original_labels = labels[:, :-1].squeeze(1)  # in shape (N,)

        exponents_pos = torch.zeros(len(original_labels), device=self.device)
        exponents_neg = torch.zeros(len(original_labels), device=self.device)
        exponents_pos -= self.multipliers[0] * protected
        exponents_neg -= self.multipliers[1] * protected
        weights_pos = torch.sigmoid(exponents_pos) # Shape: (N,)
        weights_neg = torch.sigmoid(exponents_neg) # Shape: (N,)

        self.weights = torch.where(original_labels > 0, 1 - weights_pos, weights_neg)

    def compute_violations(self, predictions, labels):
        protected = labels[:, -1]  # Shape (N,)
        original_labels = labels[:, :-1].squeeze(1)  # Shape (N,)

        violations = torch.zeros_like(self.multipliers, device=self.device)

        num_groups = len(self.multipliers) // 2

        for i in range(num_groups):
            # Find indices for the current protected group
            protected_group = (protected == i)
            positive_label = (original_labels == 1)

            # Ensure that protected_group and positive_label are the same shape
            protected_group = protected_group.unsqueeze(1)
            positive_label = positive_label.unsqueeze(1)

            # Compute violations for positive labels
            pos_group_preds = predictions[protected_group & positive_label]
            if pos_group_preds.numel() > 0:
                violations[2 * i] = (pos_group_preds != 1).float().sum()

            # Compute violations for negative labels
            neg_group_preds = predictions[protected_group & ~positive_label]
            if neg_group_preds.numel() > 0:
                violations[2 * i + 1] = (neg_group_preds != 0).float().sum()

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
                all_predictions.append(outputs.detach())
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
            self.debias_weights(all_labels)
            violations = self.compute_violations(all_predictions, all_labels)
            self.multipliers += self.eta * violations
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
