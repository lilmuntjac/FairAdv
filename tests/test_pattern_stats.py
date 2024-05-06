import re
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

class TestPatternStats(unittest.TestCase):
    """
    A unittest class for verifying the pattern performance with saved statistics.
    This test ensures that the pattern performance, when re-evaluated, matches previously computed 
    results stored in the training statistics.
    """

    @classmethod
    def setUpClass(cls):
        setup_start = time.perf_counter()
        config_path = './config/testing/cb_at_m_ps.yml'  # Specify the path to your configuration file
        cls.config = load_config(config_path)
        cls.device = cls.config_device(cls.config)
        # Load the base model and the dataset first
        cls.model_type = cls.config['dataset'].get('type', 'binary')
        model = cls.setup_model_class(cls.config, cls.device)
        cls.model = cls.load_model_checkpoint(cls.config, model, cls.device)
        _, cls.val_loader = cls.select_data_loader(cls.config)
        # Load the pattern (into applier) and its stats
        cls.applier, cls.pattern_stats = cls.load_pattern_and_stats(cls.config, cls.device)
        cls.epoch = cls.config['unit_test'].get('epoch', -1)
        if cls.epoch < 0: # no epoch is detected, get the epoch by the checkpoints name
            cls.epoch = cls.extract_epoch_number(cls.config['unit_test']['pattern_load_path'])
            print(f"Automatically detect the epoch number {cls.epoch} from the file name.")
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
    def extract_epoch_number(path):
        load_path = Path(path).name
        match = re.search(r'(\d+)(?=\.\w+$)', load_path)
        return int(match.group(1)) if match else -1
    
    @staticmethod
    def setup_model_class(config, device):
        training_schema = config['dataset'].get('training_schema', '')
        if training_schema in ['generic', 'pattern', 'mfd']:
            model = GenericModel(num_subgroups=config['dataset']['num_subgroups']).to(device)
        elif training_schema in ['contrastive',]:
            model = GenericModel(contrastive=True).to(device)
        else:
            raise ValueError(f"Invalid training method: {training_schema}")
        return model
    
    @staticmethod
    def load_model_checkpoint(config, model, device):
        load_path = Path(config['unit_test']['model_load_path'])
        if not load_path.exists():
            raise FileNotFoundError(f"Model checkpoint file does not exist at specified path: {load_path}")
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model
    
    @staticmethod
    def load_pattern_and_stats(config, device):
        pattern_type = config['unit_test']['pattern_type']
        pattern_path = Path(config['unit_test']['pattern_load_path'])
        frame_thickness = config['unit_test'].get('frame_thickness', 0.05)
        if not pattern_path.exists():
            raise FileNotFoundError(f"Pattern checkpoint file does not exist at specified path: {pattern_path}")
        # Select and return the appropriate applier
        if pattern_type == 'perturbation':
            applier = PerturbationApplier(perturbation_path=pattern_path, device=device)
        elif pattern_type == 'frame':
            applier = FrameApplier(frame_thickness=frame_thickness, frame_path=pattern_path, device=device)
        elif pattern_type == 'eyeglasses':
            applier = EyeglassesApplier(eyeglasses_path=pattern_path, device=device)
        else:
            raise ValueError(f"Invalid pattern type: {pattern_type}")

        pattern_stats_path = Path(config['unit_test']['pattern_load_stats'])
        if not pattern_stats_path.exists():
            raise FileNotFoundError(f"Pattern stats file does not exist at specified path: {pattern_stats_path}")
        loaded_pattern_stats = torch.load(pattern_stats_path)

        return applier, loaded_pattern_stats
    
    @staticmethod
    def select_data_loader(config):
        dataset_name = config['dataset']['name']
        pattern_type = config.get('attack', {}).get('pattern_type', 'perturbation')

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
    
    def test_checkpoint_accuracy(self):
        new_stats = self.evaluate_model()
        loaded_stats = self.pattern_stats['val'][self.epoch - 1]
        np.testing.assert_array_almost_equal(
            new_stats, loaded_stats, decimal=6, err_msg="Stats do not match."
        )
        # If the assertion passes, print the original saved stats
        task_name = self.config['dataset'].get('task_name', 'unspecified')
        train_stats = self.pattern_stats['train'][self.epoch - 1]
        val_stats   = self.pattern_stats['val'][self.epoch - 1]
        if self.model_type == 'binary':
            print_binary_model_summary(train_stats, val_stats, task_name)
        elif self.model_type == 'multi-class':
            print_multiclass_model_summary(train_stats, val_stats, task_name)
        else:
            raise ValueError(f"Invalid dataset type: {self.model_type}")

    def compute_stats(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        if self.model_type == 'binary':
            stats = get_confusion_matrix_counts(predicted, labels)
        elif self.model_type == 'multi-class':
            stats = get_rights_and_wrongs_counts(predicted, labels)
        else:
            raise ValueError(f"Invalid dataset type: {self.model_type}")
        return stats
    
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

    def evaluate_model(self):
        self.model.eval()
        total_stats = None
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                processed_images, labels = self.embed_pattern(batch)
                processed_images = normalize(processed_images)
                outputs = self.model(processed_images)
                stats = self.compute_stats(outputs, labels)
                total_stats = stats if total_stats is None else total_stats + stats
        return total_stats

if __name__ == '__main__':
    unittest.main()
