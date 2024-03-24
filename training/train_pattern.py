import time
from pathlib import Path

import torch

import utils.utils as utils
from adversarial.losses.direct_loss import binary_eqodd_loss

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
        # applier that applies pattern onto the image
        self.applier = utils.select_applier(config, device=device)
        self.attack_params = {
            'method': config['attack']['method'],
            'pattern_type': config['attack']['pattern_type'],
            'epsilon': config['attack']['epsilon'],
            'alpha': config['attack']['alpha'],
            'iters': config['attack']['iters'],
        }
        
        self.save_path = Path(save_path) # Ensure it's a Path object
        self.device = device

    def compute_loss(self, outputs, labels):
        match (self.model_type, self.attack_params['method']):
            case ('binary', 'direct'):
                fair_loss = binary_eqodd_loss(outputs, labels)
                loss = torch.mean(fair_loss)
            case _:
                raise NotImplementedError()
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

    def embed_pattern(self, batch):
        pattern_type = self.attack_params['pattern_type']
        if pattern_type in ['perturbation', 'frame']:
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            processed_images = self.applier.apply(images)
            return processed_images, labels
        elif pattern_type == 'eyeglasses':
            images, theta, labels = batch
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
            loss = self.compute_loss(outputs, labels)
            loss.backward()
            self.applier.update(self.attack_params['alpha'], self.attack_params['epsilon'])
            stats = self.compute_stats(outputs, labels)
            # update the total loss and total stats for this epoch
            total_loss += loss.item()
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
        if start_epoch >= final_epoch:
            print("Start epoch must be less than final epoch.")
            return
        total_start_time = time.perf_counter()
        if start_epoch == 1:
            # train from scratch, get stats for epoch 0 first
            _, init_train_stats = self.validate_epoch(use_trainloader=True)
            _, init_val_stats   = self.validate_epoch(use_trainloader=False)
            total_train_stats = torch.cat((total_train_stats, init_train_stats.unsqueeze(0)), dim=0)
            total_val_stats   = torch.cat((total_val_stats,   init_val_stats.unsqueeze(0)), dim=0)
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

