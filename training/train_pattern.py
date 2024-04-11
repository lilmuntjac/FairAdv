import time
from pathlib import Path

import torch

import utils.utils as utils
from adversarial.losses import BinaryDirectLoss, BinaryMaskingLoss, BinaryPerturbedLoss

class FairPatternTrainer:
    def __init__(self, config, train_loader, val_loader, 
                 model, criterion, optimizer, scheduler, save_path, device='cpu'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.model_type = config['dataset']['type']
        self.attr_list = config['dataset'].get('selected_attrs', [])
        self.attr_name = config['dataset'].get('selected_attr', 'unspecified')
        self.criterion = criterion # unused
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_path = Path(save_path) # Ensure it's a Path object
        self.device = device
        # applier that applies pattern onto the image
        self.applier = utils.select_applier(config, device=device)
        self.attack_params = {
            'pattern_type': config['attack']['pattern_type'],
            'method': config['attack']['method'],
            'alpha': config['attack']['alpha'],
            'iters': config['attack']['iters'],
            'epsilon': config['attack'].get('epsilon', 0), # for perturbation
            'gamma': config['attack'].get('gamma', 0),     # balancing fairness and accuracy
            'gamma_adjust_factor': config['attack'].get('gamma_adjust_factor', 0),
            'ratio_history': [],
            'ratio_history_length': 10,
        }
        self.load_pretrained_weight(config)
        self.get_fairness_loss(config)

    def load_pretrained_weight(self, config):
        # load pre-trained model weights
        model_path = Path(config['model']['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"The pre-trained model weight file does not exist: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def get_fairness_loss(self, config):
        model_type = config['dataset'].get('type', 'binary')
        loss_type = config['attack'].get('method', 'direct')
        fairness_criteria = config['attack'].get('fairness_criteria', 'equalized odds')
        match (model_type, loss_type):
            case ('binary', 'direct'):
                self.fairness_loss = BinaryDirectLoss(fairness_criteria=fairness_criteria)
            case ('binary', 'masking'):
                self.fairness_loss = BinaryMaskingLoss(fairness_criteria=fairness_criteria)
            case ('binary', 'perturbed'):
                self.fairness_loss = BinaryPerturbedLoss(fairness_criteria=fairness_criteria)
            case _:
                raise NotImplementedError()
            
    def compute_loss(self, outputs, labels):
        loss = self.fairness_loss.compute_loss(outputs, labels)
        # may add code that adjust parameter based on current fairness status
        return loss

    def compute_stats(self, outputs, labels):
        if self.model_type == 'binary':
            predicted = (outputs > 0.5).float()
            stats = utils.get_confusion_matrix_counts(predicted, labels)
        elif self.model_type == 'multi-class':
            _, predicted = torch.max(outputs, 1)
            stats = utils.get_rights_and_wrongs_counts(predicted, labels)
        else:
            raise ValueError(f"Invalid dataset type: {self.model_type}")
        return stats

    def show_stats(self, train_stats, val_stats):
        if self.model_type == 'binary':
            utils.print_binary_model_summary(train_stats, val_stats, self.attr_list)
        elif self.model_type == 'multi-class':
            utils.print_multiclass_model_summary(train_stats, val_stats, self.attr_name)
        else:
            raise ValueError(f"Invalid dataset type: {self.model_type}")
        
    def adjust_gamma_based_on_gradient(self, fairness_grad, accuracy_grad):
        """
        Adjusts the gamma based on the gradient magnitudes of fairness and accuracy losses.

        Args:
            fairness_grad (torch.Tensor): Gradient of the fairness loss.
            accuracy_grad (torch.Tensor): Gradient of the accuracy loss.
        """
        threshold_ratio, min_gamma, max_gamma = 3.0, 0, 100
        fairness_grad_norm = fairness_grad.norm()
        accuracy_grad_norm = accuracy_grad.norm()

        # 
        # fairness_grad_flat = fairness_grad.view(-1)
        # accuracy_grad_flat = accuracy_grad.view(-1)
        # cosine_similarity = torch.dot(fairness_grad_flat, accuracy_grad_flat) / (fairness_grad_norm * accuracy_grad_norm)

        # Adjust gamma based on the ratio of gradient norms
        if fairness_grad_norm > 0 and accuracy_grad_norm > 0:
            current_ratio  = accuracy_grad_norm / fairness_grad_norm

            # Update the ratio history
            self.attack_params['ratio_history'].append(current_ratio)
            if len(self.attack_params['ratio_history']) > self.attack_params['ratio_history_length']:
                self.attack_params['ratio_history'].pop(0)  # Remove the oldest ratio to maintain the defined history length
            avg_ratio = sum(self.attack_params['ratio_history']) / len(self.attack_params['ratio_history'])

            if avg_ratio > threshold_ratio:
                self.attack_params['gamma'] += self.attack_params['gamma_adjust_factor']
            else:
                self.attack_params['gamma'] -= self.attack_params['gamma_adjust_factor']
            self.attack_params['gamma'] = min(max_gamma, max(min_gamma, self.attack_params['gamma']))
            # print(f'fairness: {fairness_grad_norm:.4f} / accuracy: {accuracy_grad_norm:.4f} / sim: {cosine_similarity:.4f}')
            # print(f"Ratio: {avg_ratio:.4f} Adjusted gamma: {self.attack_params['gamma']:.4f}")

    def embed_pattern(self, batch):
        pattern_type = self.attack_params['pattern_type']
        if pattern_type in ['perturbation', 'frame']:
            images, labels, *_ = batch
            images, labels = images.to(self.device), labels.to(self.device)
            processed_images = self.applier.apply(images)
            return processed_images, labels
        elif pattern_type == 'eyeglasses':
            images, theta, labels, *_ = batch
            images, theta, labels = images.to(self.device), theta.to(self.device), labels.to(self.device)
            processed_images = self.applier.apply(images, theta)
            return processed_images, labels

    # Train the pattern for one epoch.
    def train_epoch(self):
        self.model.eval() # train the pattern, not model
        total_loss, total_stats = 0, None
        for batch_idx, batch in enumerate(self.train_loader):
            processed_images, labels = self.embed_pattern(batch)
            processed_images = utils.normalize(processed_images)
            outputs = self.model(processed_images)

            # Calculate losses
            fairness_loss = self.fairness_loss.compute_loss(outputs, labels)
            accuracy_loss = self.fairness_loss.compute_utlity_loss(outputs, labels)

            # Get the gradients from both losses
            self.applier.clear_gradient()
            fairness_loss.backward(retain_graph=True)
            fairness_grad = self.applier.get_gradient().clone()
            self.applier.clear_gradient()
            accuracy_loss.backward(retain_graph=True)
            accuracy_grad = self.applier.get_gradient().clone()
            # Adjust gamma based on the gradient magnitudes
            self.adjust_gamma_based_on_gradient(fairness_grad, accuracy_grad)
        
            # Compute total loss with updated gamma and update parameters
            self.applier.clear_gradient()
            total_loss_value = fairness_loss + self.attack_params['gamma'] * accuracy_loss
            total_loss_value.backward()
            self.applier.update(self.attack_params['alpha'], self.attack_params['epsilon'])
            
            stats = self.compute_stats(outputs, labels)
            # update the total loss and total stats for this epoch
            total_loss += total_loss_value.item()
            total_stats = stats if total_stats is None else total_stats + stats
        avg_loss = total_loss / (batch_idx + 1)
        return avg_loss, total_stats
    
    # Validate the pattern for one epoch.
    def validate_epoch(self, use_trainloader=False):
        dataloader = self.train_loader if use_trainloader else self.val_loader
        self.model.eval()
        total_loss, total_stats = 0, None
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                processed_images, labels = self.embed_pattern(batch)
                processed_images = utils.normalize(processed_images)
                outputs = self.model(processed_images)
                loss = self.compute_loss(outputs, labels)
                stats = self.compute_stats(outputs, labels)
                # update the total loss and total stats for this epoch
                total_loss += loss.item()
                total_stats = stats if total_stats is None else total_stats + stats
            avg_loss = total_loss / (batch_idx + 1)
        return avg_loss, total_stats

    def run(self, start_epoch, final_epoch, total_train_stats, total_val_stats):
        if start_epoch > final_epoch:
            print("Start epoch must be less than final epoch.")
            return
        total_start_time = time.perf_counter()
        if start_epoch == 1:
            # train from scratch, get stats for epoch 0 first
            _, init_train_stats = self.validate_epoch(use_trainloader=True)
            _, init_val_stats   = self.validate_epoch(use_trainloader=False)
            total_train_stats = torch.cat((total_train_stats, init_train_stats.unsqueeze(0)), dim=0)
            total_val_stats   = torch.cat((total_val_stats,   init_val_stats.unsqueeze(0)), dim=0)
            print(f'Load model with the stats:')
            self.show_stats(init_train_stats, init_val_stats)

        for epoch in range(start_epoch, final_epoch + 1):
            epoch_start_time = time.perf_counter()
            train_loss, train_stats = self.train_epoch()
            val_loss,   val_stats   = self.validate_epoch()
            if self.scheduler:
                self.scheduler.step()
            epoch_time = time.perf_counter() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds ({epoch_time / 60:.2f} minutes)")
            print(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            self.show_stats(train_stats, val_stats)
            print('-'*100)
            # Concatenate the new epoch's stats along the epoch dimension
            total_train_stats = torch.cat((total_train_stats, train_stats.unsqueeze(0)), dim=0)
            total_val_stats   = torch.cat((total_val_stats,   val_stats.unsqueeze(0)), dim=0)
            # Save the optimized pattern per epoch
            pattern_path = self.save_path / f'pattern_epoch_{epoch:04d}.pt'
            torch.save(self.applier.get(), pattern_path)
        # Save the training stats
        stats_path = self.save_path / f'stats_end_{final_epoch:04d}.pt'
        torch.save({
            'epoch': final_epoch,
            'train': total_train_stats,
            'val'  : total_val_stats,
        }, stats_path)
        total_time = time.perf_counter() - total_start_time
        print(f"Total Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

