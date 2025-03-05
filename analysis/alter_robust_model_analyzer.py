import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Add the project root to the Python path
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent  # Adjust this path to point to the project root
sys.path.append(str(project_root))

import torch

from models.generic_model import GenericModel
from data.loaders.dataloader import (create_celeba_data_loaders, create_celeba_xform_data_loaders,
                                     create_fairface_data_loaders, create_fairface_xform_data_loaders,
                                     create_ham10000_data_loaders)
from utils.config_utils import load_config, config_env
from utils.metrics_utils import (
    normalize, get_confusion_matrix_counts, get_rights_and_wrongs_counts, 
    print_binary_model_summary, print_multiclass_model_summary
)
from adversarial import PerturbationApplier, FrameApplier, EyeglassesApplier

class AlterRobustModelAnalyser:
    def __init__(self, config, device='cpu'):
        _, self.val_loader = self.setup_dataloader(config)
        self.model = self.setup_model(config, device)
        self.model_type = config['dataset']['type']
        self.pattern_type = config['analysis'].get('load_pattern_type', 'perturbation')
        self.applier = self.setup_applier(config, device)
        self.task_name = config['dataset'].get('task_name', 'unspecified')
        self.criterion = self.setup_criterion(config) 
        self.device = device
        # adversarial attck related hyperparameter
        self.batch_size = config['analysis'].get('batch_size', 64)
        self.adv_ratio = config['attack'].get('adv_ratio', 1.0)
        self.alpha = config['attack'].get('alpha', 0)
        self.iters = config['attack'].get('iters', 0)
        self.epsilon = config['attack'].get('epsilon', 0)

    def setup_dataloader(self, config):
        dataset_name = config['dataset']['name']
        pattern_type = config['analysis'].get('load_pattern_type', 'perturbation')

        # Loader function with or without a transformation matrix.
        if pattern_type == 'eyeglasses':
            if dataset_name == 'celeba':
                loader_function = create_celeba_xform_data_loaders
            elif dataset_name == 'fairface':
                loader_function = create_fairface_xform_data_loaders
            elif dataset_name == 'ham10000':
                raise ValueError('Cannot apply eyeglasses onto HAM10000 dataset')
        else:
            # Default behavior for 'perturbation' and 'frame'
            if dataset_name == 'celeba':
                loader_function = create_celeba_data_loaders
            elif dataset_name == 'fairface':
                loader_function = create_fairface_data_loaders
            elif dataset_name == 'ham10000':
                loader_function = create_ham10000_data_loaders
                train_loader, val_loader = loader_function(batch_size=config['analysis']['batch_size'])
                return train_loader, val_loader
            else:
                raise ValueError(f"Invalid configuration: dataset={dataset_name}, pattern={pattern_type}")

        balanced = config['dataset'].get('balanced', False)
        # Configure the dataloader based on the training method. 
        if not balanced:
            train_loader, val_loader = loader_function(
                selected_attrs=config['dataset']['selected_attrs'] + [config['dataset']['protected_attr']],
                batch_size=config['analysis']['batch_size']
            )
        else:
            train_loader, val_loader = loader_function(
                selected_attrs=config['dataset']['selected_attrs'] + [config['dataset']['protected_attr']],
                batch_size=config['analysis']['batch_size'],
                sampler='balanced_batch_sampler' # BalancedBatchSampler
            )
        return train_loader, val_loader
    
    def setup_model(self, config, device):
        # Setup the pre-trained model classes
        model_path = Path(config['analysis']['load_model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"The pre-trained model weight file does not exist: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model = GenericModel(num_outputs=config['dataset']['num_outputs']).to(device)
        # load the model weight
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def setup_criterion(self, config):
        criterion_type = config['dataset'].get('type', 'binary')
        if criterion_type == 'binary':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif criterion_type == 'multi-class':
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion type: {criterion_type}")
        return criterion
    
    def setup_applier(self, config, device):
        # Load the pre-trained pattern
        pattern_path = Path(config['analysis']['load_pattern_path'])
        if not pattern_path.exists():
            raise FileNotFoundError(f"Pattern checkpoint file does not exist at specified path: {pattern_path}")
        pattern_type = config['analysis']['load_pattern_type']
        frame_thickness = config['analysis']['load_frame_thickness']
        # Select and return the appropriate applier
        if pattern_type == 'perturbation':
            applier = PerturbationApplier(perturbation_path=pattern_path, device=device)
        elif pattern_type == 'frame':
            applier = FrameApplier(frame_thickness=frame_thickness, frame_path=pattern_path, device=device)
        elif pattern_type == 'eyeglasses':
            applier = EyeglassesApplier(eyeglasses_path=pattern_path, device=device)
        else:
            raise ValueError(f"Invalid pattern type: {pattern_type}")
        return applier
    
    def pgd_attack(self, images, labels, theta=None, include_applier=False):
        is_training = self.model.training  # Save the original mode of the model (train or eval)
        
        # add adversarial noise
        # random initial point for pgd attack
        noise_images = images.clone().detach().to(self.device)
        random_noise = torch.empty_like(noise_images).uniform_(-self.epsilon, self.epsilon).to(self.device)
        noise_images = torch.clamp(noise_images + random_noise, min=0, max=1).detach_()
        adv_images = noise_images.clone().detach().to(self.device)
        # only attack on the corrected predicted instances
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
        # attack loop
        adv_images.requires_grad = True
        for _ in range(self.iters):
            if include_applier:
                if self.pattern_type in ['perturbation', 'frame']:
                    adv_images.data = self.applier.apply(adv_images)
                elif self.pattern_type == 'eyeglasses':
                    adv_images.data = self.applier.apply(adv_images, theta)

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
    
    def embed_pattern(self, batch):
        if self.pattern_type in ['perturbation', 'frame']:
            images, labels, *_ = batch
            images, labels = images.to(self.device), labels.to(self.device)
            processed_images = self.applier.apply(images)
            return processed_images, labels
        elif self.pattern_type == 'eyeglasses':
            images, theta, labels, *_ = batch
            images, theta, labels = images.to(self.device), theta.to(self.device), labels.to(self.device)
            processed_images = self.applier.apply(images, theta)
            return processed_images, labels
    
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
    
    # Validate the robust accuracy for one epoch.
    def validate_robust_accuracy(self, attack_with_pattern=True):
        self.model.eval()
        total_loss, total_stats = 0, None
        for batch_idx, batch in enumerate(self.val_loader):
            # unpack the batch of data based on the pattern type
            if self.pattern_type in ['perturbation', 'frame']:
                images, labels, *_ = batch
                images, labels = images.to(self.device), labels.to(self.device)
                theta = None
            elif self.pattern_type == 'eyeglasses':
                images, theta, labels, *_ = batch
                images, theta, labels = images.to(self.device), theta.to(self.device), labels.to(self.device)
            
            adv_images = self.pgd_attack(images=images, labels=labels, theta=theta, include_applier=attack_with_pattern)
            # put on the pattern before send into the model
            if self.pattern_type in ['perturbation', 'frame']:
                adv_images = self.applier.apply(adv_images)
            elif self.pattern_type == 'eyeglasses':
                adv_images = self.applier.apply(adv_images, theta)
            adv_outputs = self.model(normalize(adv_images))
            loss = self.compute_loss(adv_outputs, labels)
            stats = self.compute_stats(adv_outputs, labels)
            # update the total loss and total stats for this epoch
            total_loss += loss.item()
            total_stats = stats if total_stats is None else total_stats + stats
        avg_loss = total_loss / (batch_idx + 1)
        return avg_loss, total_stats
    
    def run(self):
        total_start_time = time.perf_counter()
        val_robust_loss_n, val_robust_stats_n = self.validate_robust_accuracy(attack_with_pattern=False)
        val_robust_loss_p, val_robust_stats_p = self.validate_robust_accuracy(attack_with_pattern=True)
        print(f'Validation Loss (N): {val_robust_loss_n:.4f}, Validation Loss (P): {val_robust_loss_p:.4f}')
        self.show_stats(val_robust_stats_n, val_robust_stats_p)

        total_time = time.perf_counter() - total_start_time
        print(f"Total Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")


def main(config):
    device = config_env(config, title='analysis')
    analyser = AlterRobustModelAnalyser(config, device)
    analyser.run()

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config_path', type=str, help='Path to the configuration YAML file')

    # Load the configuration file specified by the command-line argument
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config)