import time
from pathlib import Path

import torch
import torch.nn as nn

import utils.utils as utils

class GenericTrainer:
    def __init__(self, config, train_loader, val_loader, 
                 model, criterion, optimizer, scheduler, save_path, device='cpu'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.model_type = config['dataset']['type']
        self.attr_list = config['dataset'].get('selected_attrs', [])
        self.attr_name = config['dataset'].get('selected_attr', 'unspecified')
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_path = Path(save_path) # Ensure it's a Path object
        self.device = device

    def compute_loss(self, outputs, labels):
        if self.model_type == 'binary':
            loss = self.criterion(outputs, labels[:, :-1])
        elif self.model_type == 'multi-class':
            loss = self.criterion(outputs, labels[:, :-1].squeeze(1))
        else:
            raise ValueError(f"Invalid dataset type: {self.model_type}")
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

    # Train the model for one epoch.
    def train_epoch(self):
        self.model.train()
        total_loss, total_stats = 0, None
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            images = utils.normalize(images)
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
                images = utils.normalize(images)
                outputs = self.model(images)
                loss = self.compute_loss(outputs, labels)
                stats = self.compute_stats(outputs, labels)
                # update the total loss and total stats for this epoch
                total_loss += loss.item()
                total_stats = stats if total_stats is None else total_stats + stats
            avg_loss = total_loss / (batch_idx + 1)
        return avg_loss, total_stats

    def run(self, start_epoch, final_epoch, total_train_stats, total_val_stats):
        if start_epoch >= final_epoch:
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
            print('-'*100)
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
