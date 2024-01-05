import yaml

import torch

from .data_loader import (create_celeba_data_loaders, create_celeba_xform_data_loaders,
                          create_fairface_data_loaders, create_fairface_xform_data_loaders)
from adversarial import PerturbationApplier, FrameApplier, EyeglassesApplier

def load_config(config_path):
    """ Load YAML configuration file. """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def set_seed(seed):
    """ Set the seed for reproducibility in PyTorch and CUDA environments. """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def select_data_loader(config):
    dataset_name = config['dataset']['name']
    pattern_type = config['attack']['pattern_type']

    if pattern_type in ['perturbation', 'frame']:
        if dataset_name == 'celeba':
            return create_celeba_data_loaders
        elif dataset_name == 'fairface':
            return create_fairface_data_loaders

    elif pattern_type == 'eyeglasses':
        if dataset_name == 'celeba':
            return create_celeba_xform_data_loaders
        elif dataset_name == 'fairface':
            return create_fairface_xform_data_loaders

    raise ValueError(f"Invalid configuration: dataset={dataset_name}, pattern={pattern_type}")

def select_applier(config, device='cpu'):
    pattern_type = config['attack']['pattern_type']
    base_path  = config['attack'].get('base_path')

    if pattern_type == 'perturbation':
        if base_path:
            return PerturbationApplier(perturbation_path=base_path, device=device)
        else:
            # Random tensor for PerturbationApplier
            epsilon = config['attack']['epsilon']
            random_tensor = torch.rand((1, 3, 224, 224)) * 2 - 1
            random_tensor.clamp_(-epsilon, epsilon)
            return PerturbationApplier(perturbation=random_tensor, device=device)
    elif pattern_type == 'frame':
        frame_thickness = config['attack'].get('frame_thickness', 0.1)
        if base_path:
            return FrameApplier(frame_thickness=frame_thickness, frame_path=base_path, device=device)
        else:
            # Random tensor for FrameApplier
            random_tensor = torch.rand((1, 3, 224, 224))
            return FrameApplier(frame_thickness=frame_thickness, frame=random_tensor, device=device)
    elif pattern_type == 'eyeglasses':
        if base_path:
            return EyeglassesApplier(eyeglasses_path=base_path, device=device)
        else:
            # Random tensor for EyeglassesApplier
            random_tensor = torch.rand((1, 3, 224, 224))
            return EyeglassesApplier(eyeglasses=random_tensor, device=device)
    else:
        raise ValueError(f"Invalid pattern: {pattern_type}")

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