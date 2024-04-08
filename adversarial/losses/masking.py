import torch
import torch.nn.functional as F

from .base_binary import BinaryBaseLoss

class BinaryMaskingLoss(BinaryBaseLoss):
    def __init__(self, fairness_criteria):
        super().__init__(fairness_criteria)

    def _compute_fairness_loss(self, outputs, labels):
        """
        Computes the masking fairness loss based on the specified fairness criteria.

        Args:
            outputs (torch.Tensor): Model predictions.
            labels (torch.Tensor): Ground truth labels with protected attributes.

        Returns:
            torch.Tensor: Calculated fairness loss.
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
        match self.fairness_criteria:
            case 'equality of opportunity':
                masks[(protected_attribute == lower_tpr_group).unsqueeze(-1) & fn_mask] = 1
                masks[(protected_attribute == lower_tpr_group).unsqueeze(-1) & tp_mask] = 1
            case 'equalized odds':
                masks[(protected_attribute == lower_tpr_group).unsqueeze(-1) & fn_mask] = 1
                masks[(protected_attribute == lower_tpr_group).unsqueeze(-1) & tp_mask] = 1
                masks[(protected_attribute == higher_fpr_group).unsqueeze(-1) & fp_mask] = 1
                masks[(protected_attribute == higher_fpr_group).unsqueeze(-1) & tn_mask] = 1
            case _:
                raise NotImplementedError(f"Fairness criteria '{self.fairness_criteria}' is not supported.")

        generic_loss = F.binary_cross_entropy(outputs, actual_labels, reduction='none')
        masking_loss = (generic_loss * masks).sum() / masks.sum()
        return masking_loss.mean()