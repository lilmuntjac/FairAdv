import yaml

import torch

def load_config(config_path):
    """ Load YAML configuration file. """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def set_seed(seed):
    """ Set the seed for reproducibility in PyTorch and CUDA environments. """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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