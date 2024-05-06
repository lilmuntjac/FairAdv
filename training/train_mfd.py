import time
from pathlib import Path

import torch
import torch.nn as nn

from utils.metrics_utils import (
    normalize, get_confusion_matrix_counts, get_rights_and_wrongs_counts, 
    print_binary_model_summary, print_multiclass_model_summary
)

class MFDTrainer:
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
        # MFD related hyperparameter
        self.mmd_criterion = MMDLoss(device=device).to(device)
        self.lamda = config['mfd'].get('lamda', 1)
        self.teacher = self.get_teacher_model(config).to(device)

    def get_teacher_model(self, config):
        # Infer the number of output features/classes from self.model's final layer
        if hasattr(self.model, 'classification_head'):  # Assuming the final layer is named 'fc'
            num_outputs = self.model.classification_head.out_features
        else:
            raise AttributeError("Fail to get output dimension from self.model")
        
        teacher_model_class = self.model.__class__
        teacher_model = teacher_model_class(num_outputs).to(self.device)

        # Load teacher model weights
        teacher_path = Path(config['mfd']['teacher_path'])
        if not teacher_path.exists():
            raise FileNotFoundError(f"The teacher model weight file does not exist: {teacher_path}")
        checkpoint = torch.load(teacher_path, map_location=self.device)
        teacher_model.load_state_dict(checkpoint['model_state_dict'])
        return teacher_model

    def compute_stats(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        if self.model_type == 'binary':
            stats = get_confusion_matrix_counts(predicted, labels)
        elif self.model_type == 'multi-class':
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

    def train_epoch(self):
        self.teacher.eval() # teacher model
        self.model.train()  # student model
        total_loss, total_stats = 0, None
        for batch_idx, (images, labels, subgroups) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            images = normalize(images)
            # the original model training loss
            self.optimizer.zero_grad()
            outputs, student_feature = self.model(images, get_feature=True)
            generic_loss = loss = self.criterion(outputs, labels[:,0])
            # MMD loss
            _, teacher_feature = self.teacher(images, get_feature=True)
            mmd_loss = self.mmd_criterion(teacher_feature, student_feature, subgroups)
            loss = generic_loss + self.lamda * mmd_loss
            loss.backward()
            self.optimizer.step()
            stats = self.compute_stats(outputs, labels)
            # update the total loss and total stats for this epoch
            total_loss += loss.item()
            total_stats = stats if total_stats is None else total_stats + stats
        avg_loss = total_loss / (batch_idx + 1)
        return avg_loss, total_stats

    def validate_epoch(self):
        self.teacher.eval() # teacher model
        self.model.eval()  # student model
        total_loss, total_stats = 0, None
        # So far, the validation set is the original (without balancing), so no support in MMD loss.
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                images = normalize(images)
                # the original model training loss
                outputs, student_feature = self.model(images, get_feature=True)
                # generic_loss = loss = self.criterion(outputs, labels[:,0])
                # MMD loss
                # _, teacher_feature = self.teacher(images, get_feature=True)
                # mmd_loss = self.mmd_criterion(teacher_feature, student_feature, subgroups)
                # loss = generic_loss + self.lamda * mmd_loss
                stats = self.compute_stats(outputs, labels)
                # update the total loss and total stats for this epoch
                # total_loss += loss.item()
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

class MMDLoss(nn.Module):
    def __init__(self, device='cpu'):
        super(MMDLoss, self).__init__()
        self.device = device

    def forward(self, teacher_feature, student_feature, subgroups):
        teacher_feature = teacher_feature.reshape(teacher_feature.size(0), -1)
        student_feature = student_feature.reshape(student_feature.size(0), -1)

        squared_distances = self.compute_squared_distances(teacher_feature, student_feature)
        bandwidth_multiplier = torch.sqrt(squared_distances.mean()).detach()

        if not isinstance(subgroups, torch.Tensor):
            subgroups = torch.stack(subgroups, dim=1).to(self.device)
        else:
            subgroups = subgroups.to(self.device)
        unique_subgroups = subgroups.unique(dim=0)
        # Compute RBF kernel matrices
        mmd_loss = 0.0
        for current_subgroup in unique_subgroups:
            subgroup_indices = (subgroups == current_subgroup).all(dim=1) # same target, same protected
            attributes_indices = (subgroups[:, :-1] == current_subgroup[:-1]).all(dim=1) # same target
            
            teacher_subgroup_features = teacher_feature[attributes_indices]
            student_subgroup_features = student_feature[subgroup_indices]
            # Compute RBF kernel matrices for current subgroup
            K_TS = self.compute_rbf_kernel(teacher_subgroup_features, student_subgroup_features, bandwidth_multiplier=bandwidth_multiplier)
            K_SS = self.compute_rbf_kernel(student_subgroup_features, student_subgroup_features, bandwidth_multiplier=bandwidth_multiplier)
            K_TT = self.compute_rbf_kernel(teacher_subgroup_features, teacher_subgroup_features, bandwidth_multiplier=bandwidth_multiplier)
            # Compute MMD loss for this subgroup
            mmd_loss += K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()

        return mmd_loss
        
    def compute_squared_distances(self, embeddings1, embeddings2):
        """ Computes the squared Euclidean distances between two sets of vectors. """
        eps = 1e-12 # A small value to ensure numerical stability.
        squares_embeddings1 = torch.sum(embeddings1 ** 2, dim=1).view(-1, 1)
        squares_embeddings2 = torch.sum(embeddings2 ** 2, dim=1).view(1, -1)
        dot_product = torch.matmul(embeddings1, embeddings2.t())
        squared_distances = squares_embeddings1 + squares_embeddings2 - 2 * dot_product
        squared_distances = squared_distances.clamp(min=eps)
        # The matrix of squared distances of shape (num_samples_1, num_samples_2)
        return squared_distances

    def compute_rbf_kernel(self, embeddings1, embeddings2, bandwidth=1.0, bandwidth_multiplier=None):
        """ Computes the RBF (Gaussian) kernel between two sets of vectors """
        squared_distances = self.compute_squared_distances(embeddings1, embeddings2)

        if bandwidth_multiplier is None:
            bandwidth = torch.sqrt(squared_distances.mean())
        
        # The computed RBF kernel matrix of shape (num_samples_1, num_samples_2)
        rbf_kernel_matrix = torch.exp(-squared_distances / (2 * bandwidth_multiplier * bandwidth))
        return rbf_kernel_matrix

        