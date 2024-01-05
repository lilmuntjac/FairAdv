import torch
import torch.nn.functional as F

def binary_eqodd_loss(outputs, labels):
    """
    Compute a loss to minimize the difference in equalized odds between groups.

    Args:
        outputs (torch.Tensor): The model's predictions.
        labels (torch.Tensor): The true labels along with the protected attribute as the last column.

    Returns:
        torch.Tensor: The computed loss.
    """
    protected_attrs = labels[:, -1]
    actual_labels = labels[:, :-1]  # Exclude the protected attribute
    

    group1_mask = (protected_attrs == 0)
    group2_mask = (protected_attrs == 1)
    outputs_group1, labels_group1 = outputs[group1_mask], actual_labels[group1_mask]
    outputs_group2, labels_group2 = outputs[group2_mask], actual_labels[group2_mask]

    # Differentiable approximation to a step function
    steep_sigmoid = lambda x: 1. / (1. + torch.exp(-1e2 * (x - 0.5)))

    # List to store loss for each attribute
    attr_losses = []
    # Iterate over each attribute
    for attr_idx in range(actual_labels.shape[1]):
        outputs_attr_group1 = steep_sigmoid(outputs_group1[:, attr_idx])
        outputs_attr_group2 = steep_sigmoid(outputs_group2[:, attr_idx])
        labels_attr_group1 = labels_group1[:, attr_idx]
        labels_attr_group2 = labels_group2[:, attr_idx]

        # Calculate TPR and FPR for each group
        tpr_group1 = (outputs_attr_group1 * labels_attr_group1).sum() / (labels_attr_group1.sum() + 1e-8)
        tpr_group2 = (outputs_attr_group2 * labels_attr_group2).sum() / (labels_attr_group2.sum() + 1e-8)
        fpr_group1 = (outputs_attr_group1 * (1 - labels_attr_group1)).sum() / ((1 - labels_attr_group1).sum() + 1e-8)
        fpr_group2 = (outputs_attr_group2 * (1 - labels_attr_group2)).sum() / ((1 - labels_attr_group2).sum() + 1e-8)

        # Loss to minimize the absolute difference in TPR and FPR between groups for each attribute
        tpr_loss = torch.abs(tpr_group1 - tpr_group2)
        fpr_loss = torch.abs(fpr_group1 - fpr_group2)

        # Add attribute-specific loss to the list
        attr_losses.append(tpr_loss + fpr_loss)

    # Convert list of losses to a tensor
    return torch.stack(attr_losses)