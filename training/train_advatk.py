import time
from pathlib import Path

import torch
import torch.nn.functional as F

from utils.metrics_utils import (
    normalize, get_confusion_matrix_counts, get_rights_and_wrongs_counts, 
    print_binary_model_summary, print_multiclass_model_summary
)

class AdvTrainer:
    def __init__(self, config, train_loader, val_loader,
                 model, criterion, optimizer, scheduler, save_path, device='cpu'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.model_type = config['dataset']['type']
        self.task_name = config['dataset'].get('task_name', 'unspecified')
        self.criterion = criterion 
        self.optimizer = optimizer 
        self.scheduler = scheduler
        self.save_path = Path(save_path) # Ensure it's a Path object
        self.device = device
        # adversarial attck related hyperparameter
        self.batch_size = config['training'].get('batch_size', 64)
        self.adv_ratio = config['attack'].get('adv_ratio', 1.0)
        self.alpha = config['attack'].get('alpha', 0)
        self.iters = config['attack'].get('iters', 0)
        self.epsilon = config['attack'].get('epsilon', 0)
        
    def pgd_attack(self, images, labels):
        is_training = self.model.training  # Save the original mode of the model (train or eval)
        
        noise_images = images.clone().detach().to(self.device)
        random_noise = torch.empty_like(noise_images).uniform_(-self.epsilon, self.epsilon).to(self.device)
        noise_images = torch.clamp(noise_images + random_noise, min=0, max=1).detach_()
        adv_images = noise_images.clone().detach().to(self.device)

        self.model.eval()
        outputs = self.model(normalize(adv_images))
        if self.model_type == 'binary':
            # The binary model does not include a sigmoid function in the output. 
            # Remember to pass the output through the sigmoid function first, 
            # then obtain the predictions from it.
            outputs = torch.sigmoid(outputs)
            predicted = (outputs > 0.5).int()
            incorrect_indices = (predicted != labels[:, :-1].float()).squeeze()
        else:
            _, predicted = torch.max(outputs, 1)
            incorrect_indices = (predicted != labels[:, :-1].squeeze(1)).squeeze()

        adv_images.requires_grad = True
        for _ in range(self.iters):
            outputs = self.model(normalize(adv_images))
            loss = self.compute_loss(outputs, labels)
            loss.backward()

            # Non-targeted attack
            with torch.no_grad():
                adv_images = adv_images + self.alpha * adv_images.grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach_()

            adv_images.requires_grad = True
        adv_images.requires_grad = False
        adv_images[incorrect_indices] = noise_images[incorrect_indices]
        if is_training:
            self.model.train()
        return adv_images
    
    def compute_loss(self, outputs, labels):
        if self.model_type == 'binary':
            loss = self.criterion(outputs, labels[:, :-1].float())
        elif self.model_type == 'multi-class':
            loss = self.criterion(outputs, labels[:, :-1].squeeze(1))
        else:
            raise ValueError(f"Invalid dataset type: {self.model_type}")
        return loss

    def compute_weighted_loss(self, outputs, labels, is_adversarial):
        # Get the standard classification loss:
        if self.model_type == 'binary':
            base_loss = self.criterion(outputs, labels[:, :-1].float())
        elif self.model_type == 'multi-class':
            base_loss = self.criterion(outputs, labels[:, :-1].squeeze(1))
        else:
            raise ValueError(f"Invalid dataset type: {self.model_type}")
        
        # Calculate confidence penalty for adversarial examples
        confidence_penalty = 0.0
        # if is_adversarial.sum() > 0:  # Only compute if there are adversarial examples in the batch
        #     adv_outputs = outputs[is_adversarial == 1]  # Select adversarial examples
        #     if self.model_type == 'binary':
        #         probs = torch.sigmoid(adv_outputs)
        #         confidence_penalty = -torch.mean(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
        #     elif self.model_type == 'multi-class':
        #         probs = F.softmax(adv_outputs, dim=1)
        #         confidence_penalty = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))

        # 0.1 is a hyperparameter for regularization strength
        # Experiment with different values (e.g., 0.05, 0.1, 0.2)
        return base_loss + 0.1 * confidence_penalty

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
    def train_epoch(self, adv_ratio=0.9):
        assert 0.0 <= adv_ratio <= 1.0, "adv_ratio must be between 0.0 and 1.0"
        self.model.train()
        total_loss, total_stats = 0, None
        for batch_idx, (images, labels, *_) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            adv_images = self.pgd_attack(images, labels)

            num_adv = int(self.batch_size * adv_ratio)
            indices = torch.randperm(self.batch_size)

            mixed_images = torch.cat((images[indices[:self.batch_size - num_adv]], adv_images[indices[:num_adv]]))
            mixed_labels = torch.cat((labels[indices[:self.batch_size - num_adv]], labels[indices[:num_adv]]))
            is_adversarial = torch.cat((torch.zeros(self.batch_size - num_adv), torch.ones(num_adv))).to(self.device)
            # combine_images = normalize(torch.cat((images, adv_images), dim=0))
            # combine_labels = torch.cat((labels, labels), dim=0)

            self.optimizer.zero_grad()
            outputs = self.model(mixed_images)
            loss = self.compute_weighted_loss(outputs, mixed_labels, is_adversarial)
            loss.backward()
            self.optimizer.step()
            stats = self.compute_stats(outputs, mixed_labels)
            # update the total loss and total stats for this epoch
            total_loss += loss.item()
            total_stats = stats if total_stats is None else total_stats + stats
        avg_loss = total_loss / (batch_idx + 1)
        return avg_loss, total_stats
    
    # Validate the model for one epoch.
    def validate_epoch(self):
        self.model.eval()
        total_loss, total_stats = 0, None
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                images = normalize(images)
                outputs = self.model(images)
                loss = self.compute_loss(outputs, labels)
                stats = self.compute_stats(outputs, labels)
                # update the total loss and total stats for this epoch
                total_loss += loss.item()
                total_stats = stats if total_stats is None else total_stats + stats
            avg_loss = total_loss / (batch_idx + 1)
        return avg_loss, total_stats
    
    # Validate the robust accuracy for one epoch.
    def validate_robust_accuracy(self):
        self.model.eval()
        total_loss, total_stats = 0, None
        for batch_idx, (images, labels) in enumerate(self.val_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            adv_images = self.pgd_attack(images, labels)
            adv_outputs = self.model(normalize(adv_images))
            loss = self.compute_loss(adv_outputs, labels)
            stats = self.compute_stats(adv_outputs, labels)
            # update the total loss and total stats for this epoch
            total_loss += loss.item()
            total_stats = stats if total_stats is None else total_stats + stats
        avg_loss = total_loss / (batch_idx + 1)
        return avg_loss, total_stats

    def run(self, start_epoch, final_epoch, total_train_stats, total_val_stats):
        total_val_robust_stats = total_val_stats.clone() # do not support reload yet
        if start_epoch > final_epoch:
            print("Start epoch must be less than final epoch.")
            return
        total_start_time = time.perf_counter()
        for epoch in range(start_epoch, final_epoch + 1):
            epoch_start_time = time.perf_counter()
            train_loss, train_stats           = self.train_epoch(self.adv_ratio)
            val_loss, val_stats               = self.validate_epoch()
            val_robust_loss, val_robust_stats = self.validate_robust_accuracy()
            if self.scheduler:
                self.scheduler.step()
            epoch_time = time.perf_counter() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds ({epoch_time / 60:.2f} minutes)")
            print(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            self.show_stats(train_stats, val_stats)
            self.show_stats(train_stats, val_robust_stats)
            print('-'*120)
            # Concatenate the new epoch's stats along the epoch dimension
            total_train_stats = torch.cat((total_train_stats, train_stats.unsqueeze(0)), dim=0)
            total_val_stats   = torch.cat((total_val_stats,   val_stats.unsqueeze(0)), dim=0)
            total_val_robust_stats = torch.cat((total_val_robust_stats, val_robust_stats.unsqueeze(0)), dim=0)
            # Save model checkpoint per epoch
            checkpoint_path = self.save_path / f'checkpoint_epoch_{epoch:04d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
            }, checkpoint_path)
        # Save the training stats
        stats_path = self.save_path / f'stats_end_{final_epoch:04d}.pt'
        torch.save({
            'epoch': final_epoch,
            'train': total_train_stats,
            'val'  : total_val_stats,
            'val_robust': total_val_robust_stats,
        }, stats_path)
        total_time = time.perf_counter() - total_start_time
        print(f"Total Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")