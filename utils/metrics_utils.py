import torch

def normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Normalize a batch of images using the specified mean and standard deviation.
    Args:
    - images (Tensor): Batch of images.
    - mean (list): Per-channel mean to subtract.
    - std (list): Per-channel standard deviation to divide.

    Returns:
    - Normalized images as a tensor.
    """
    mean = torch.tensor(mean, device=images.device).view(-1, 1, 1)
    std = torch.tensor(std, device=images.device).view(-1, 1, 1)
    return (images - mean) / std

def get_confusion_matrix_counts(predictions, labels):
    """
    Calculate confusion matrix counts for each of two groups determined by a protected attribute. 
    Computes TP, FP, FN, TN for each group based on binary predictions and a binary main attribute.

    Args:
    - predictions: A PyTorch tensor of model predictions.
    - labels: A PyTorch tensor of true labels, where the last column is the protected attribute.

    Returns:
    - conf_matrix: A PyTorch tensor of shape (2, 4), each row corresponds to a group defined
                   by the protected attribute, containing [TP, FP, FN, TN].
    """
    conf_matrix = torch.zeros((2, 4), dtype=torch.int64)  # Initialize the confusion matrix tensor

    protected_attr = labels[:, -1]  # Extract the protected attribute

    for group_idx in [0, 1]:
        group_mask = (protected_attr == group_idx)
        group_preds = predictions[group_mask, 0] # Assume there's only 1 target attribute
        group_labels = labels[group_mask, 0] # Assume labels for the targeted attribute are in the first column

        TP = torch.sum((group_preds == 1) & (group_labels == 1))
        FP = torch.sum((group_preds == 1) & (group_labels == 0))
        FN = torch.sum((group_preds == 0) & (group_labels == 1))
        TN = torch.sum((group_preds == 0) & (group_labels == 0))

        conf_matrix[group_idx] = torch.tensor([TP, FP, FN, TN], dtype=torch.int64)

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
        group_labels = labels[group_mask, 0]  # Exclude the protected attribute

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

def print_binary_model_summary(train_stats_count, val_stats_count, task_name):
    """
    Print a summary of binary model training and validation metrics.

    Args:
    - train_stats_count (Tensor): Training statistics tensor of shape (2, 4),
                                  where 2 represents two groups and 4 cells in the confusion matrix.
    - val_stats_count (Tensor): Validation statistics tensor of shape (2, 4),
                                similar to train_stats.
    - task_name (str): Name of the task or attribute being evaluated.
    """
    train_group1_matrix = train_stats_count[0]
    train_group2_matrix = train_stats_count[1]
    val_group1_matrix   = val_stats_count[0]
    val_group2_matrix   = val_stats_count[1]
        
    train_metrics = calculate_metrics_for_attribute(train_group1_matrix, train_group2_matrix)
    val_metrics   = calculate_metrics_for_attribute(val_group1_matrix, val_group2_matrix)

    print(f'\nAttribute {task_name} Metrics:')
    print(f'  Train      - Group 1 Accuracy: {train_metrics[0]:.4f}, Group 2 Accuracy: {train_metrics[1]:.4f}, '
          f'Total Accuracy: {train_metrics[2]:.4f}, Equalized Odds: {train_metrics[3]:.4f}')
    print(f'  Validation - Group 1 Accuracy: {val_metrics[0]:.4f}, Group 2 Accuracy: {val_metrics[1]:.4f}, '
          f'Total Accuracy: {val_metrics[2]:.4f}, Equalized Odds: {val_metrics[3]:.4f}')
        
def print_multiclass_model_summary(train_stats_count, val_stats_count, task_name):
    """
    Print a summary of multiclass model training and validation metrics.

    Args:
    - train_stats_count (Tensor): Tensor of training statistics where each row represents a group 
                                  and contains counts of correct and incorrect predictions.
    - val_stats_count (Tensor): Tensor of validation statistics similar to training statistics.
    - task_name (str): Name of the task or attribute being evaluated.
    """
    train_group1_matrix = train_stats_count[0]
    train_group2_matrix = train_stats_count[1]
    val_group1_matrix   = val_stats_count[0]
    val_group2_matrix   = val_stats_count[1]

    train_metrics = calculate_accuracy_for_attribute(train_group1_matrix, train_group2_matrix)
    val_metrics   = calculate_accuracy_for_attribute(val_group1_matrix, val_group2_matrix)

    print(f'\nAttribute {task_name} Metrics:')
    print(f'  Train      - Group 1 Accuracy: {train_metrics[0]:.4f}, Group 2 Accuracy: {train_metrics[1]:.4f}, '
          f'Total Accuracy: {train_metrics[2]:.4f}, Acc. diffrernce: {train_metrics[3]:.4f}')
    print(f'  Validation - Group 1 Accuracy: {val_metrics[0]:.4f}, Group 2 Accuracy: {val_metrics[1]:.4f}, '
          f'Total Accuracy: {val_metrics[2]:.4f}, Acc. diffrernce: {val_metrics[3]:.4f}')
