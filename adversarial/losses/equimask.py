import torch
import torch.nn.functional as F

from .binary_base import BinaryLossBase

class BinaryEquimask(BinaryLossBase):
    def __init__(self, fairness_criteria):
        """
        Initializes the BinaryEquimask loss class
        """
        super().__init__(fairness_criteria)

    def _compute_loss(self, outputs, labels):
        """
        Computes the equimask loss according to the specified fairness criteria. The method applies
        fairness adjustments by selectively penalizing parts of the standard cross-entropy loss
        based on the fairness criterion.

        Args:
            outputs (torch.Tensor): The logits from a model.
            labels (torch.Tensor): Ground truth labels which include protected attributes.

        Returns:
            torch.Tensor: The computed fairness-adjusted loss.
        """
        # Extract the protected attribute and actual labels from the labels tensor
        protected_attribute = labels[:, -1] # in shape (N,)
        actual_labels = labels[:, 0]        # in shape (N,)
        _, predictions = torch.max(outputs, 1) # in shape (N,)

        # Split predictions and labels by protected attribute
        group_0_pred,  group_1_pred  = self._split_by_protected_attribute(predictions, protected_attribute)
        group_0_label, group_1_label = self._split_by_protected_attribute(actual_labels, protected_attribute)

        # Calculate TPR and FPR for both groups
        group_0_tpr = self._get_tpr(group_0_pred, group_0_label)
        group_1_tpr = self._get_tpr(group_1_pred, group_1_label)
        group_0_fpr = self._get_fpr(group_0_pred, group_0_label)
        group_1_fpr = self._get_fpr(group_1_pred, group_1_label)

        # Initialize masks to adjust loss based on fairness considerations
        masks = torch.zeros_like(predictions)

        lower_tpr_group,  higher_tpr_group  = (0, 1) if group_0_tpr < group_1_tpr else (1, 0)
        higher_fpr_group, lower_fpr_group   = (0, 1) if group_0_fpr > group_1_fpr else (1, 0)

        tp_mask = ((predictions == 1) & (actual_labels == 1))
        fp_mask = ((predictions == 1) & (actual_labels == 0))
        fn_mask = ((predictions == 0) & (actual_labels == 1))
        tn_mask = ((predictions == 0) & (actual_labels == 0))

        # Apply masks based on the specified fairness criteria
        match self.fairness_criteria:
            case 'equality of opportunity':
                masks[(protected_attribute == lower_tpr_group) & fn_mask] = 1
                masks[(protected_attribute == lower_tpr_group) & tp_mask] = 1
            case 'equalized odds':
                masks[(protected_attribute == lower_tpr_group) & fn_mask] = 1
                masks[(protected_attribute == lower_tpr_group) & tp_mask] = 1
                masks[(protected_attribute == higher_fpr_group) & fp_mask] = 1
                masks[(protected_attribute == higher_fpr_group) & tn_mask] = 1
            case _:
                raise NotImplementedError(f"Fairness criteria '{self.fairness_criteria}' is not supported.")

        # Calculate the masked cross entropy loss, considering only selected parts based on fairness
        cross_entropy = F.cross_entropy(outputs, actual_labels, reduction='none')
        mask_count = masks.sum()
        loss = (cross_entropy * masks).sum() / (mask_count if mask_count > 0 else 1)
        return loss.mean()
