import os
import yaml

import torch

from adversarial import PerturbationApplier, FrameApplier, EyeglassesApplier
from data.loaders.dataloader import (create_celeba_data_loaders, create_celeba_xform_data_loaders,
                                     create_fairface_data_loaders, create_fairface_xform_data_loaders,
                                     create_ham10000_data_loaders)
from training.train_generic import GenericTrainer
from training.train_pattern import FairPatternTrainer
from training.train_mfd import MFDTrainer

def load_config(config_path):
    """ Load YAML configuration file. """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def config_env(config):
    """ Configures the device for PyTorch and initializes random seeds. """
    print(f"PyTorch Version: {torch.__version__}")

    # Set the CUDA devices based on the configuration.
    use_cuda = config['training'].get('use_cuda', False)
    gpu_id = config['training'].get('gpu_id', 0)
    gpu_setting = ",".join(map(str, gpu_id)) if isinstance(gpu_id, list) else str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_setting

    # Check if CUDA is available after setting CUDA_VISIBLE_DEVICES.
    if use_cuda and not torch.cuda.is_available():
        print(f"GPU setting: {gpu_setting}")
        raise RuntimeError("CUDA requested but not available. Please check your GPU settings.")
    elif use_cuda:
        if isinstance(gpu_id, list):
            print(f"Using CUDA devices: GPUs {gpu_setting}")
        else:
            print(f"Using CUDA device: GPU {gpu_setting}")
    else:
        print("Using CPU for computations.")

    # Set the computation device based on CUDA availability.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize random seeds for PyTorch to ensure reproducibility.
    seed = config['training'].get('random_seed', 2665)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return device

def select_model(config):
    if config['dataset']['name'] in ['celeba', 'fairface']:
        model = BinaryModel(len(config['dataset']['selected_attrs']))
        criterion = nn.BCELoss()
    elif config['dataset']['name'] == 'ham10000':
        class_num = config['dataset']['class_number']
        model = MulticlassModel(class_num)
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid dataset name: {config['dataset']['name']}")
    return model, criterion

def select_data_loader(config):
    dataset_name = config['dataset']['name']
    pattern_type = config.get('attack', {}).get('pattern_type', 'perturbation')
    training_method = config.get('training', {}).get('method', 'generic')

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
            train_loader, val_loader = loader_function(batch_size=config['training']['batch_size'])
            return train_loader, val_loader
        else:
            raise ValueError(f"Invalid configuration: dataset={dataset_name}, pattern={pattern_type}")
    # Configure the dataloader based on the training method. 
    if training_method == 'generic': 
        train_loader, val_loader = loader_function(
            selected_attrs=config['dataset']['selected_attrs'] + [config['dataset']['protected_attr']],
            batch_size=config['training']['batch_size']
        )
    elif training_method == 'mfd':
        train_loader, val_loader = loader_function(
            selected_attrs=config['dataset']['selected_attrs'] + [config['dataset']['protected_attr']],
            batch_size=config['training']['batch_size'],
            sampler='balanced_batch_sampler' # BalancedBatchSampler
        )
    return train_loader, val_loader

def select_trainer(config):
    trainer_type = config['training'].get('method')
    if trainer_type == 'generic':
        return GenericTrainer
    elif trainer_type == 'pattern':
        return FairPatternTrainer
    elif trainer_type =='mfd':
        return MFDTrainer
    else:
        raise ValueError(f"Invalid trainer method specified in config: {trainer_type}")

def select_applier(config, pattern=None, device='cpu'):
    pattern_type = config['attack']['pattern_type']
    base_path  = config['attack'].get('base_path')
    epsilon = config['attack'].get('epsilon')
    frame_thickness = config['attack'].get('frame_thickness')

    # Create or use the provided pattern
    if pattern is None and not base_path:
        random_tensor = torch.rand((1, 3, 224, 224)) * 2 - 1
        if pattern_type == 'perturbation':
            pattern = random_tensor * 2 - 1
            pattern = pattern.clamp_(-epsilon, epsilon)
        elif pattern_type in ['frame', 'eyeglasses']:
            pattern = random_tensor

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

def normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """ Normalize a batch of images. """
    mean = torch.tensor(mean, device=images.device).view(-1, 1, 1)
    std = torch.tensor(std, device=images.device).view(-1, 1, 1)
    return (images - mean) / std

def get_confusion_matrix_counts(predictions, labels):
    """
    Calculate confusion matrix counts for each attribute, split by protected attribute.
    
    Args:
    - predictions: A PyTorch tensor of model predictions.
    - labels: A PyTorch tensor of true labels, where the last column is the protected attribute.

    Returns:
    - A PyTorch tensor of shape (A, 2, 4), where A is the number of attributes excluding 
      the protected attribute. Each sub-tensor contains the confusion matrix (TP, FP, FN, TN)
      for a specific attribute and group.
    """
    num_attrs = labels.shape[1] - 1  # Number of attributes excluding the protected attribute
    conf_matrix = torch.zeros((num_attrs, 2, 4), dtype=torch.int64)  # Initialize the confusion matrix tensor

    protected_attr = labels[:, -1]  # Extract the protected attribute

    for attr_idx in range(num_attrs):
        for group_idx in [0, 1]:
            group_mask = (protected_attr == group_idx)

            group_preds = predictions[group_mask][:, attr_idx]
            group_labels = labels[group_mask][:, attr_idx]

            TP = torch.sum((group_preds == 1) & (group_labels == 1))
            FP = torch.sum((group_preds == 1) & (group_labels == 0))
            FN = torch.sum((group_preds == 0) & (group_labels == 1))
            TN = torch.sum((group_preds == 0) & (group_labels == 0))

            conf_matrix[attr_idx, group_idx] = torch.tensor([TP, FP, FN, TN])

    return conf_matrix

def get_rights_and_wrongs_counts(predictions, labels):
    """
    Calculate the counts of correct and incorrect predictions for each attribute, 
    split by the protected attribute.

    Args:
    - predictions (torch.Tensor): A tensor of model predictions.
    - labels (torch.Tensor): A tensor of true labels, where the last column 
                             represents the protected attribute.

    Returns:
    - torch.Tensor: A tensor of shape (2, 2), where the first dimension represents 
                    the group based on the protected attribute, and the second 
                    dimension contains counts of correct (index 0) and incorrect 
                    (index 1) predictions.
    """
    counts = torch.zeros((2, 2), dtype=torch.int64)  # Tensor to store counts

    protected_attr = labels[:, -1]

    for group_idx in [0, 1]:
        group_mask = (protected_attr == group_idx)
        group_preds = predictions[group_mask]
        group_labels = labels[group_mask,0]  # Exclude the protected attribute

        rights = torch.sum(group_preds == group_labels)
        wrongs = torch.sum(group_preds != group_labels)

        counts[group_idx] = torch.tensor([rights, wrongs], dtype=torch.int64)

    return counts

def calculate_accuracy(TP, FP, FN, TN):
    """ Calculate accuracy given confusion matrix components. """
    total = TP + FP + FN + TN
    if total == 0:
        return 0
    return (TP + TN) / total

def calculate_equalized_odds(TP1, FP1, FN1, TN1, TP2, FP2, FN2, TN2):
    """ Calculate equalized odds. Adjust this formula based on your definition of equalized odds. """
    # For instance, equalized odds could be calculated as the average of absolute differences in FPR and TPR
    TPR1, TPR2 = TP1 / (TP1 + FN1), TP2 / (TP2 + FN2)
    FPR1, FPR2 = FP1 / (FP1 + TN1), FP2 / (FP2 + TN2)
    return abs(TPR1 - TPR2) + abs(FPR1 - FPR2)

def calculate_metrics_for_attribute(group1_matrix, group2_matrix):
    """ Calculate metrics for a single attribute. """
    TP1, FP1, FN1, TN1 = group1_matrix
    TP2, FP2, FN2, TN2 = group2_matrix

    group1_acc = calculate_accuracy(TP1, FP1, FN1, TN1)
    group2_acc = calculate_accuracy(TP2, FP2, FN2, TN2)
    total_acc = calculate_accuracy(TP1 + TP2, FP1 + FP2, FN1 + FN2, TN1 + TN2)
    equalized_odds = calculate_equalized_odds(TP1, FP1, FN1, TN1, TP2, FP2, FN2, TN2)

    return group1_acc, group2_acc, total_acc, equalized_odds

def calculate_accuracy_for_attribute(group1_matrix, group2_matrix):
    """ Calculate accuracy for a single attribute. """
    R1, W1 = group1_matrix
    R2, W2 = group2_matrix

    group1_acc = R1 / (R1 + W1) if R1 + W1 != 0 else 0
    group2_acc = R2 / (R2 + W2) if R2 + W2 != 0 else 0
    total_instances = R1 + R2 + W1 + W2
    total_acc = (R1 + R2) / total_instances if total_instances != 0 else 0
    differences_acc = abs(group1_acc-group2_acc)

    return group1_acc, group2_acc, total_acc, differences_acc

def print_binary_model_summary(train_stats_count, val_stats_count, attr_list):
    for attr_index, attr_name in enumerate(attr_list):
        # Extract confusion matrices for the current attribute and epoch
        train_group1_matrix = train_stats_count[attr_index, 0]
        train_group2_matrix = train_stats_count[attr_index, 1]
        val_group1_matrix = val_stats_count[attr_index, 0]
        val_group2_matrix = val_stats_count[attr_index, 1]
        
        train_metrics = calculate_metrics_for_attribute(train_group1_matrix, train_group2_matrix)
        val_metrics = calculate_metrics_for_attribute(val_group1_matrix, val_group2_matrix)
        print(f'\nAttribute {attr_name} Metrics:')
        print(f'  Train      - Group 1 Accuracy: {train_metrics[0]:.4f}, Group 2 Accuracy: {train_metrics[1]:.4f}, '
              f'Total Accuracy: {train_metrics[2]:.4f}, Equalized Odds: {train_metrics[3]:.4f}')
        print(f'  Validation - Group 1 Accuracy: {val_metrics[0]:.4f}, Group 2 Accuracy: {val_metrics[1]:.4f}, '
              f'Total Accuracy: {val_metrics[2]:.4f}, Equalized Odds: {val_metrics[3]:.4f}')
        
def print_multiclass_model_summary(train_stats_count, val_stats_count, attr_name) :
    train_group1_matrix = train_stats_count[0]
    train_group2_matrix = train_stats_count[1]
    val_group1_matrix = val_stats_count[0]
    val_group2_matrix = val_stats_count[1]

    train_metrics = calculate_accuracy_for_attribute(train_group1_matrix, train_group2_matrix)
    val_metrics = calculate_accuracy_for_attribute(val_group1_matrix, val_group2_matrix)

    print(f'\nAttribute {attr_name} Metrics:')
    print(f'  Train      - Group 1 Accuracy: {train_metrics[0]:.4f}, Group 2 Accuracy: {train_metrics[1]:.4f}, '
          f'Total Accuracy: {train_metrics[2]:.4f}, Acc. diffrernce: {train_metrics[3]:.4f}')
    print(f'  Validation - Group 1 Accuracy: {val_metrics[0]:.4f}, Group 2 Accuracy: {val_metrics[1]:.4f}, '
          f'Total Accuracy: {val_metrics[2]:.4f}, Acc. diffrernce: {val_metrics[3]:.4f}') 