import os
import sys
import time
import unittest
import numpy as np
from pathlib import Path

# Add the project root to the Python path
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent  # Adjust this path to point to the project root
sys.path.append(str(project_root))

import torch
import kornia
from torchvision.utils import save_image

from models.generic_model import GenericModel
from data.loaders.dataloader import (create_celeba_data_loaders, create_celeba_xform_data_loaders,
                                     create_fairface_data_loaders, create_fairface_xform_data_loaders,
                                     create_ham10000_data_loaders)
from utils.config_utils import load_config
from utils.metrics_utils import (
    normalize, get_confusion_matrix_counts, get_rights_and_wrongs_counts, 
    print_binary_model_summary, print_multiclass_model_summary
)
from adversarial import PerturbationApplier, FrameApplier, EyeglassesApplier

class TestPattern(unittest.TestCase):
    """
    A unittest class for testing the application of various adversarial patterns
    (perturbations, frames, eyeglasses) to images from different datasets.
    Please check the patterned images manually
    """

    @classmethod
    def setUpClass(cls):
        setup_start = time.perf_counter()
        config_path = './config/testing/cb_at_ptn.yml'  # Specify the path to your configuration file
        cls.config = load_config(config_path)
        cls.device = cls.config_device(cls.config)
        # Load the dataset first
        cls.train_loader, cls.val_loader = cls.select_data_loader(cls.config)
        # Load the pattern (into applier)
        cls.pattern_type = cls.config['unit_test'].get('pattern_type', 'perturbation')
        cls.applier = cls.load_pattern(cls.config, cls.device)
        cls.save_path = Path(cls.config['unit_test'].get('save_path', './analysis'))
        cls.save_path.mkdir(parents=True, exist_ok=True)
        cls.eyeglasses_with_augment = cls.config['unit_test'].get('eyeglasses_with_augment', False)
        if cls.pattern_type == 'eyeglasses':
            cls.aug_in_grad = kornia.augmentation.container.AugmentationSequential(
                kornia.augmentation.auto.TrivialAugment()
                # kornia.augmentation.auto.RandAugment(n=2, m=10)
            ).to(cls.device)
        setup_end = time.perf_counter()
        print(f"Setup Time: {setup_end - setup_start:.4f} seconds")

    @staticmethod
    def config_device(config):
        print(f"PyTorch Version: {torch.__version__}")
        use_cuda = config['unit_test'].get('use_cuda', False)
        gpu_setting = str(config['unit_test'].get('gpu_setting', '0'))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_setting

        if use_cuda:
            print(f"GPU setting: {gpu_setting}")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is requested but not available; "
                       "check CUDA_VISIBLE_DEVICES or GPU availability.")
            device = 'cuda'
        else:
            device = 'cpu'
            print("Using CPU for computations.")

        # Initialize random seeds for PyTorch to ensure reproducibility.
        seed = config['unit_test'].get('random_seed', 2665)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)

        return device

    @staticmethod
    def load_pattern(config, device):
        pattern_type = config['unit_test']['pattern_type']
        pattern_path = Path(config['unit_test']['pattern_load_path'])
        frame_thickness = config['unit_test'].get('frame_thickness', 0.05)
        if not pattern_path.exists():
            raise FileNotFoundError(f"Pattern checkpoint file does not exist at specified path: {pattern_path}")
        # Select and return the appropriate applier
        if pattern_type  == 'perturbation':
            applier = PerturbationApplier(perturbation_path=pattern_path, device=device)
        elif pattern_type == 'frame':
            applier = FrameApplier(frame_thickness=frame_thickness, frame_path=pattern_path, device=device)
        elif pattern_type == 'eyeglasses':
            applier = EyeglassesApplier(eyeglasses_path=pattern_path, device=device)
        else:
            raise ValueError(f"Invalid pattern type: {pattern_type}")
        return applier

    @staticmethod
    def select_data_loader(config):
        dataset_name = config['dataset']['name']
        pattern_type = config['unit_test'].get('pattern_type', 'perturbation')

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
                train_loader, val_loader = loader_function(batch_size=config['unit_test']['batch_size'])
                return train_loader, val_loader
            else:
                raise ValueError(f"Invalid configuration: dataset={dataset_name}, pattern={pattern_type}")

        balanced = config['dataset'].get('balanced', False)
        # Configure the dataloader based on the training method. 
        if not balanced:
            train_loader, val_loader = loader_function(
                selected_attrs=config['dataset']['selected_attrs'] + [config['dataset']['protected_attr']],
                batch_size=config['unit_test']['batch_size']
            )
        else:
            train_loader, val_loader = loader_function(
                selected_attrs=config['dataset']['selected_attrs'] + [config['dataset']['protected_attr']],
                batch_size=config['unit_test']['batch_size'],
                sampler='balanced_batch_sampler' # BalancedBatchSampler
            )
        return train_loader, val_loader

    def test_pattern(self):
        # Save images from the train dataset
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.train_loader):
                processed_images, labels = self.embed_pattern(batch)
                if self.pattern_type == 'eyeglasses' and self.eyeglasses_with_augment:
                    print(f'  Do augmentation for eyeglasses')
                    processed_images = self.aug_in_grad(processed_images)
                # Save the first 16 images in the batch
                for i in range(min(16, processed_images.size(0))):
                    image_path = self.save_path / f'train_{i:02}.png'
                    save_image(processed_images[i], image_path)
                break  # Only run for the first batch
        # Save images from the validation dataset
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                processed_images, labels = self.embed_pattern(batch)
                # Save the first 16 images in the batch
                for i in range(min(16, processed_images.size(0))):
                    image_path = self.save_path / f'val_{i:02}.png'
                    save_image(processed_images[i], image_path)
                break  # Only run for the first batch

    def embed_pattern(self, batch):
        pattern_type = self.config['unit_test'].get('pattern_type', 'perturbation')
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
