import time
from pathlib import Path

import torch
import kornia

from utils.metrics_utils import (
    normalize, get_confusion_matrix_counts, get_rights_and_wrongs_counts, 
    print_binary_model_summary, print_multiclass_model_summary
)
from adversarial import PerturbationApplier, FrameApplier, EyeglassesApplier
from adversarial.losses.binary_combinedloss import CombinedBinaryLoss

class FairPatternTrainer:
    def __init__(self, config, train_loader, val_loader, 
                 model, criterion, optimizer, scheduler, save_path, device='cpu'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.model_type = config['dataset']['type']
        self.task_name = config['dataset'].get('task_name', 'unspecified')
        self.criterion = criterion # unused
        self.optimizer = optimizer # unused
        self.scheduler = scheduler
        self.save_path = Path(save_path) # Ensure it's a Path object
        self.device = device
        # applier that applies pattern onto the image
        # If an existing pattern is used, load it from here.
        self.applier = self.select_applier(config, device=device)
        self.attack_params = {
            'fairness_criteria': config['attack']['fairness_criteria'],
            'pattern_type': config['attack']['pattern_type'],
            'method': config['attack']['method'],
            'alpha': config['attack']['alpha'],
            'iters': config['attack']['iters'],
            'epsilon': config['attack'].get('epsilon', 0), # for perturbation
            'gamma': config['attack'].get('gamma', 0),     # balancing fairness and accuracy
            'gamma_adjustment': config['attack'].get('gamma_adjustment', 'constant'),
            'gamma_adjust_factor': config['attack'].get('gamma_adjust_factor', 0),
            'accuracy_goal': config['attack'].get('accuracy_goal', 0.8),
            # 'ratio_history': [],
            # 'ratio_history_length': 10,
        }
        self.load_pretrained_weight(config)
        self.get_training_loss(config)
        if config['attack']['pattern_type'] == 'eyeglasses':
            trivial_augment = kornia.augmentation.auto.TrivialAugment()
            self.aug_in_grad = kornia.augmentation.AugmentationSequential(trivial_augment)

    def select_applier(self, config, pattern=None, device='cpu'):
        """
        Selects and initializes the appropriate pattern applier based on the pattern type in config.
        Args:
        - config (dict): Configuration dictionary specifying the type of pattern and other options.
        - pattern (Tensor, optional): Predefined pattern tensor; if None, a new one is generated.
        - device (str): Device of the pattern tensor.

        Returns:
        - An instance of the pattern applier.
        """
        pattern_type = config['attack']['pattern_type']
        base_path  = config['attack'].get('base_path')
        frame_thickness = config['attack'].get('frame_thickness')

        # Create or use the provided pattern
        if pattern is None and not base_path:
            # random_tensor = torch.rand((1, 3, 224, 224))
            random_tensor = torch.zeros((1, 3, 224, 224))
            if pattern_type == 'perturbation':
                pattern = random_tensor * 2 - 1
                pattern = pattern.clamp_(-0.00392, 0.00392) # single pixel value
            elif pattern_type in ['frame', 'eyeglasses']:
                pattern = random_tensor.clamp_(0, 0.00392)

        # Select and return the appropriate applier
        if pattern_type == 'perturbation':
            return PerturbationApplier(
                perturbation=pattern, perturbation_path=base_path, device=device
            )
        elif pattern_type == 'frame':
            return FrameApplier(
                frame_thickness=frame_thickness, frame=pattern, 
                frame_path=base_path, device=device
            )
        elif pattern_type == 'eyeglasses':
            return EyeglassesApplier(
                eyeglasses=pattern, eyeglasses_path=base_path, device=device
            )
        else:
            raise ValueError(f"Invalid pattern type: {pattern_type}")

    def load_pretrained_weight(self, config):
        # load pre-trained model weights
        model_path = Path(config['model']['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"The pre-trained model weight file does not exist: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
            
    def get_training_loss(self, config):
        model_type = config['dataset'].get('type', 'binary')
        loss_type = self.attack_params['method']

        # Default values for loss parameters
        fairness_criteria = self.attack_params['fairness_criteria']
        gamma = self.attack_params['gamma']
        gamma_adjustment = self.attack_params['gamma_adjustment']
        gamma_adjust_factor = self.attack_params['gamma_adjust_factor']
        accuracy_goal = self.attack_params['accuracy_goal']
        main_loss = 'binary cross entropy'  # Default main loss
        secondary_loss = None  # Default no secondary loss

        # Define loss mappings for easier configuration
        loss_mappings = {
            'binary cross entropy': (main_loss, secondary_loss),
            'fairness constraint': (main_loss, 'fairness constraint'),
            'perturbed fairness constraint': (main_loss, 'perturbed fairness constraint'),
            'EquiMask': ('EquiMask', secondary_loss),
            'EquiMask fairness constraint': ('EquiMask', 'fairness constraint'),
            'EquiMask perturbed fairness constraint': ('EquiMask', 'perturbed fairness constraint'),
        }

        if model_type == 'binary':
            if loss_type in loss_mappings:
                main_loss, secondary_loss = loss_mappings[loss_type]
                self.loss = CombinedBinaryLoss(
                    fairness_criteria=fairness_criteria,
                    main_loss=main_loss,
                    secondary_loss=secondary_loss,
                    gamma=gamma,
                    gamma_adjustment=gamma_adjustment,
                    gamma_adjust_factor=gamma_adjust_factor,
                    accuracy_goal=accuracy_goal,
                )
            else:
                raise NotImplementedError(f"Loss type '{loss_type}' is not implemented for binary models")
        else:
            raise NotImplementedError(f"Model type '{model_type}' is not supported")

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
            if self.attack_params['pattern_type'] == 'eyeglasses':
                processed_images = self.aug_in_grad(processed_images)
            processed_images = normalize(processed_images)
            outputs = self.model(processed_images)
            loss = self.loss(outputs, labels, self.applier)
            loss.backward()
            if self.attack_params['pattern_type'] == 'perturbation':
                self.applier.update(self.attack_params['alpha'], self.attack_params['epsilon'])
            elif self.attack_params['pattern_type'] in ['frame', 'eyeglasses']:
                self.applier.update(self.attack_params['alpha'])
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
                processed_images = normalize(processed_images)
                outputs = self.model(processed_images)
                loss = self.loss(outputs, labels, self.applier, train_mode=False)
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
