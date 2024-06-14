import torch
import torch.nn.functional as F

class BinaryLossBase:
    def __init__(self, fairness_criteria):
        # Can be "equality of opportunity" or "equalized odds"
        self.fairness_criteria = fairness_criteria

    def compute_loss(self, outputs, labels):
        """
        Template method to calculate loss based on fairness criteria.
        Subclasses should implement the _compute_loss method for specific computations.

        Args:
            outputs (torch.Tensor): The model's predictions.
            labels (torch.Tensor): The true labels, with protected attribute as the last column.

        Returns:
            torch.Tensor: The computed loss.
        """
        return self._compute_loss(outputs, labels)
    
    def _compute_loss(self, outputs, labels):
        """
        Placeholder for loss computation that must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _split_by_protected_attribute(self, tensor, protected_attribute, split_dim=0):
        """
        Utility to split a tensor into subgroups based on a binary protected attribute.

        Args:
            tensor (torch.Tensor): The tensor to split, can be model outputs or labels.
            protected_attribute (torch.Tensor): Tensor indicating group membership
            based on a binary protected attribute, with the same first dimension as `tensor`.
            split_dim (int): Dimension to use for splitting. 

        Returns:
            A tuple of tensors: (group_0_tensor, group_1_tensor) where each tensor corresponds to
            members of each group defined by the binary protected attribute.
        """
        group_0_mask = (protected_attribute == 0)
        group_1_mask = (protected_attribute == 1)

        # Apply mask to the tensor
        if split_dim == 0:
            group_0_tensor = tensor[group_0_mask]
            group_1_tensor = tensor[group_1_mask]
        elif split_dim == 1:
            # Assuming tensor.shape matches labels.shape
            group_0_tensor = tensor[:, group_0_mask]
            group_1_tensor = tensor[:, group_1_mask]
        else:
            raise NotImplementedError(f"Splitting by dimension {split_dim} is not implemented.")

        return group_0_tensor, group_1_tensor
    
    def _tp_cells(self, pred, label):
        return pred * label
    
    def _fp_cells(self, pred, label):
        return pred * (1 - label)
    
    def _fn_cells(self, pred, label):
        return (1 - pred) * label
    
    def _tn_cells(self, pred, label):
        return (1 - pred) * (1 - label)
    
    def _get_tpr(self, pred, label, split_dim=0):
        """
        Calculates the True Positive Rate (TPR) for the given predictions and labels.
        """
        numerator = torch.sum(self._tp_cells(pred, label), dim=split_dim) # TP
        denominator = torch.sum(label, dim=split_dim) # all positive labels
        # Use default values to replace division by zero situations, 
        # and the output tensor will be on the same device as the input.
        tpr = torch.full_like(denominator, fill_value=1.0, dtype=torch.float)
        tpr_mask = (denominator != 0)
        tpr[tpr_mask] = numerator[tpr_mask] / denominator[tpr_mask]
        return tpr
    
    def _get_fpr(self, pred, label, split_dim=0):
        """
        Calculates the False Positive Rate (FPR) for the given predictions and labels.
        """
        numerator = torch.sum(self._fp_cells(pred, label), dim=split_dim) # FP
        denominator = torch.sum((1 - label), dim=split_dim)
        # Use default values to replace division by zero situations, 
        # and the output tensor will be on the same device as the input.
        fpr = torch.full_like(denominator, fill_value=0.0, dtype=torch.float)
        fpr_mask = (denominator != 0)
        fpr[fpr_mask] = numerator[fpr_mask] / denominator[fpr_mask]
        return fpr

class BinaryCrossEntropy(BinaryLossBase):
    def __init__(self, fairness_criteria):
        super().__init__(fairness_criteria)

    def _compute_loss(self, outputs, labels):
        """
        Conpute the fairness constraint based on the specified fairness criteria.

        Args:
            outputs (torch.Tensor): Model predictions.
            labels (torch.Tensor): Ground truth labels with protected attributes.

        Returns:
            torch.Tensor: Calculated fairness loss.
        """
        actual_labels = labels[:, :-1].float() # in shape (N,)
        loss = F.binary_cross_entropy_with_logits(outputs, actual_labels)
        return loss
