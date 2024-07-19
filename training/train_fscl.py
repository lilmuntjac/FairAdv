import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics_utils import (
    normalize, get_confusion_matrix_counts, get_rights_and_wrongs_counts, 
    print_binary_model_summary, print_multiclass_model_summary
)

class FSCLSupConTrainer:
    def __init__(self, config, train_loader, val_loader,
                 model, criterion, optimizer, scheduler, save_path, device='cpu'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.model_type = config['dataset']['type']
        self.task_name = config['dataset'].get('task_name', 'unspecified')
        self.criterion = criterion # unused
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_path = Path(save_path) # Ensure it's a Path object
        self.device = device
        # FSCL+ SupCon related hyperparameter
        self.fsclsupcon_criterion = FSCLSupConLoss(
            temperature=config['fscl'].get('temperature', 0.1),
            device=device
        ).to(device)

    # Train the model for one epoch.
    def train_epoch(self):
        self.model.train()
        total_loss, total_stats = 0, None
        for batch_idx, (images, labels, *_) in enumerate(self.train_loader):
            # The variable "images" is a list containing two tensors, each with the shape (N, C, H, W).
            combined_images = torch.cat(images, dim=0).to(self.device) # (2N, C, H, W)
            combined_images = normalize(combined_images)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            features = self.model(combined_images, contrastive=True)
            loss = self.fsclsupcon_criterion(features, labels)
            loss.backward()
            self.optimizer.step()
            # update the total loss
            total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        return avg_loss, total_stats

    def run(self, start_epoch, final_epoch, total_train_stats, total_val_stats):
        if start_epoch > final_epoch:
            print("Start epoch must be less than final epoch.")
            return
        total_start_time = time.perf_counter()
        for epoch in range(start_epoch, final_epoch + 1):
            epoch_start_time = time.perf_counter()
            train_loss, train_stats = self.train_epoch()
            if self.scheduler:
                self.scheduler.step()
            epoch_time = time.perf_counter() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds ({epoch_time / 60:.2f} minutes)")
            print(f'Train Loss: {train_loss:.4f}')
            # Save model checkpoint per epoch
            checkpoint_path = self.save_path / f'checkpoint_epoch_{epoch:04d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
            }, checkpoint_path)
        # So far no stats
        total_time = time.perf_counter() - total_start_time
        print(f"Total Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

class FSCLSupConLoss(nn.Module):
    def __init__(self, temperature=0.1, base_temperature=0.05, device='cpu'):
        super(FSCLSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = device

    def create_label_mask(self, labels):
        """
        Creates a mask of shape (N, N) that checks if labels[i] == labels[j].

        Args:
            labels (Tensor): A tensor of shape (N, 1) containing the labels.

        Returns:
            Tensor: A mask of shape (N, N) where each element (i, j) is 1.0 
                    if labels[i] == labels[j], otherwise 0.0.
        """
        # Expand the labels tensor to enable broadcasting
        labels_i = labels.expand(-1, labels.size(0))      # Shape (N, N)
        labels_j = labels.t().expand(labels.size(0), -1)  # Shape (N, N)
        # Perform element-wise comparison to create the mask and convert to float
        mask = (labels_i == labels_j).float()
        return mask

    def _forward(self, features, labels):
        """
        The method for stabilizing this value 
        cannot accommodate situations where the denominator has too few terms.
        Args:
            features (Tensor): The features for all images (including 2 versions) coming from the model, 
                               should be in shape (2N, D), where N is the original batch size 
                               and D is the feature length.
            labels (Tensor): The labels for the images from the original batch, in shape (N, 2), 
                             where the first element is the targeted attribute 
                             and the second is the protected attribute.
        """
        protected_attribute = labels[:, -1].unsqueeze(1) # in shape (N, 1)
        actual_labels       = labels[:, :-1]             # in shape (N, 1)
        protected_mask = self.create_label_mask(protected_attribute).repeat(2, 2) # in shape (2N, 2N)
        # positive pairs but not pairs from the exactly same image
        labels_mask    = self.create_label_mask(actual_labels).repeat(2, 2)       # in shape (2N, 2N)
        labels_mask    = labels_mask.fill_diagonal_(0)
        # negative pairs with different protected attribute & pairs of exactly same images
        delete_mask      = (1 - labels_mask) * (1 - protected_mask)
        delete_mask      = delete_mask.fill_diagonal_(1)

        # Compute contrastive loss
        features = F.normalize(features, dim=-1)
        dot_contrast = torch.matmul(features, features.T) / self.temperature
        # Ensures pairs we don't want are not considered in the softmax calculation.
        dot_contrast = dot_contrast.masked_fill(delete_mask.bool(), -1e9)
        # For numerical stability in softmax calculation
        dot_max = torch.max(dot_contrast, dim=1, keepdim=True).values
        dot_contrast -= dot_max

        positive_pair_prob = labels_mask / labels_mask.sum(1, keepdim=True).clamp(min=1.0)
        log_softmax_scores = F.log_softmax(dot_contrast, dim=-1)
        scaled_temperature = self.temperature / self.base_temperature
        loss = - scaled_temperature * torch.sum(positive_pair_prob * log_softmax_scores, dim=-1).mean()
        return loss
    
    def forward(self, features, labels):
        """
        Args:
            features (Tensor): The features for all images (including 2 versions) coming from the model, 
                               should be in shape (2N, D), where N is the original batch size 
                               and D is the feature length.
            labels (Tensor): The labels for the images from the original batch, in shape (N, 2), 
                             where the first element is the targeted attribute 
                             and the second is the protected attribute.
        """
        protected_attribute = labels[:, -1].unsqueeze(1) # in shape (N, 1)
        actual_labels       = labels[:, :-1]             # in shape (N, 1)
        protected_mask = self.create_label_mask(protected_attribute).repeat(2, 2) # in shape (2N, 2N)
        # positive pairs but not pairs from the exactly same image
        labels_mask    = self.create_label_mask(actual_labels).repeat(2, 2)       # in shape (2N, 2N)
        labels_mask    = labels_mask.fill_diagonal_(0)

        # Compute contrastive loss
        features = F.normalize(features, dim=-1)
        dot_contrast = torch.matmul(features, features.T) / self.temperature
        dot_max = torch.max(dot_contrast, dim=1, keepdim=True).values
        dot_contrast -= dot_max

        fair_mask = (1 - labels_mask).fill_diagonal_(0) * protected_mask
        exp_contrast_sum = torch.sum(torch.exp(dot_contrast) * fair_mask, dim=1, keepdim=True)
        exp_contrast_sum = exp_contrast_sum.clamp(min=1e-9)

        log_prob = dot_contrast - torch.log(exp_contrast_sum)

        labels_mask_sum = labels_mask.sum(dim=1).clamp(min=1.0)
        mean_pos_log_prob = (labels_mask * log_prob).sum(dim=1) / labels_mask_sum
        scaled_temperature = self.temperature / self.base_temperature
        loss = - scaled_temperature * mean_pos_log_prob.mean()
        return loss

class FSCLClassifierTrainer:
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
        self.load_pretrained_weight(config)

    def load_pretrained_weight(self, config):
        # load pre-trained model weights
        model_path = Path(config['model']['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"The pre-trained model weight file does not exist: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        # Filter the state_dict to keep only the ones used in the classifier
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        # Load the new state dictionary into the model
        self.model.load_state_dict(model_dict)
        # Freeze the layers trained previously
        for name, param in self.model.named_parameters():
            if name in pretrained_dict:
                param.requires_grad = False

    def compute_loss(self, outputs, labels):
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
    def train_epoch(self):
        self.model.train()
        total_loss, total_stats = 0, None
        for batch_idx, (images, labels, *_) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            images = normalize(images)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.compute_loss(outputs, labels)
            loss.backward()
            self.optimizer.step()
            stats = self.compute_stats(outputs, labels)
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

    def run(self, start_epoch, final_epoch, total_train_stats, total_val_stats):
        if start_epoch > final_epoch:
            print("Start epoch must be less than final epoch.")
            return
        total_start_time = time.perf_counter()
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
            print('-'*120)
            # Concatenate the new epoch's stats along the epoch dimension
            total_train_stats = torch.cat((total_train_stats, train_stats.unsqueeze(0)), dim=0)
            total_val_stats   = torch.cat((total_val_stats,   val_stats.unsqueeze(0)), dim=0)
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
        }, stats_path)
        total_time = time.perf_counter() - total_start_time
        print(f"Total Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
