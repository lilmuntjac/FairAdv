import torch
import torch.nn.functional as F

class BinaryBaseLoss:
    def __init__(self, fairness_criteria):
        self.fairness_criteria = fairness_criteria

    def compute_loss(self, outputs, labels):
        """
        Template method to calculate loss. Subclasses should implement
        the _compute_fairness_loss method for specific fairness criteria.

        Args:
            outputs (torch.Tensor): The model's predictions.
            labels (torch.Tensor): The true labels along with the protected attribute as the last column.

        Returns:
            torch.Tensor: The computed loss.
        """
        fairness_loss = self._compute_fairness_loss(outputs, labels)

        return fairness_loss

    def _compute_fairness_loss(self, outputs, labels):
        """
        To be implemented by subclasses for specific fairness criteria.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def compute_utlity_loss(self, outputs, labels):
        """
        Computes the utility loss focusing on instances identified by specific masks
        within the binary classification setting.
        """
        protected_attribute = labels[:, -1]
        actual_labels = labels[:, :-1]
        predictions = torch.where(outputs > 0.5, 1, 0)

        # Split predictions and labels by protected attribute
        group_0_pred, group_1_pred = self._split_by_protected_attribute(predictions, protected_attribute)
        group_0_label, group_1_label = self._split_by_protected_attribute(actual_labels, protected_attribute)

        # Calculate TPR and FPR for both groups
        group_0_tpr = self._get_tpr(group_0_pred, group_0_label)
        group_1_tpr = self._get_tpr(group_1_pred, group_1_label)
        group_0_fpr = self._get_fpr(group_0_pred, group_0_label)
        group_1_fpr = self._get_fpr(group_1_pred, group_1_label)

        masks = torch.zeros_like(predictions)

        lower_tpr_group,  higher_tpr_group  = (0, 1) if group_0_tpr < group_1_tpr else (1, 0)
        higher_fpr_group, lower_fpr_group   = (0, 1) if group_0_fpr > group_1_fpr else (1, 0)

        tp_mask = ((predictions == 1) & (actual_labels == 1))
        fp_mask = ((predictions == 1) & (actual_labels == 0))
        fn_mask = ((predictions == 0) & (actual_labels == 1))
        tn_mask = ((predictions == 0) & (actual_labels == 0))

        # Apply masks
        masks[(protected_attribute == lower_tpr_group).unsqueeze(-1) & fn_mask] = 1
        masks[(protected_attribute == lower_tpr_group).unsqueeze(-1) & tp_mask] = 1
        # masks[(protected_attribute == higher_tpr_group).unsqueeze(-1) & tp_mask] = 1
        masks[(protected_attribute == higher_fpr_group).unsqueeze(-1) & fp_mask] = 1
        masks[(protected_attribute == higher_fpr_group).unsqueeze(-1) & tn_mask] = 1
        # masks[(protected_attribute == lower_fpr_group).unsqueeze(-1) & tn_mask] = 1

        generic_loss = F.binary_cross_entropy(outputs, actual_labels, reduction='none')
        utility_loss = (generic_loss * masks).sum() / masks.sum()
        return utility_loss.mean()

    def compute_generic_loss(self, outputs, labels):
        """
        For binary prediction model, the default training loss is binary cross entropy
        """
        actual_labels = labels[:, :-1]  # Exclude the protected attribute
        generic_loss = F.binary_cross_entropy(outputs, actual_labels, reduction='none')

        return generic_loss.mean()

        # need masking ...
        raise NotImplementedError("not finish yet you donkey.")

    def _split_by_protected_attribute(self, tensor, protected_attribute, split_dim=0):
        """
        Splits a tensor into two groups based on the protected attribute in labels.

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
    
    def _steep_sigmoid(self, x):
        """
        Applies a steep sigmoid function to the given tensor to convert raw model outputs into predictions.
        """
        return 1. / (1. + torch.exp(-1e2 * (x - 0.5)))
    
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
        fpr[fpr_mask] = numerator[fpr_mask]/denominator[fpr_mask]
        return fpr