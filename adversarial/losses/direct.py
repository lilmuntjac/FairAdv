import torch

from .base_binary import BinaryBaseLoss

class BinaryDirectLoss(BinaryBaseLoss):
    def __init__(self, fairness_criteria):
        super().__init__(fairness_criteria)

    def _compute_fairness_loss(self, outputs, labels):
        """
        Computes the direct fairness loss based on the specified fairness criteria.

        Args:
            outputs (torch.Tensor): Model predictions.
            labels (torch.Tensor): Ground truth labels with protected attributes.

        Returns:
            torch.Tensor: Calculated fairness loss.
        """
        # Convert raw outputs into predictions using a steep sigmoid function
        predictions = self._steep_sigmoid(outputs)
        
        # Split data by protected attribute
        # should be in shape (n, A) and (m, A), with n+m = N
        protected_attribute = labels[:, -1]
        group_0_pred, group_1_pred = self._split_by_protected_attribute(predictions, protected_attribute)
        group_0_label, group_1_label = self._split_by_protected_attribute(labels[:, :-1], protected_attribute)

        # Calculate fairness loss based on specified criteria
        match self.fairness_criteria:
            case 'equality of opportunity':
                # Calculate the True Positive Rate (TPR) difference between groups
                group_0_tpr = self._get_tpr(group_0_pred, group_0_label)
                group_1_tpr = self._get_tpr(group_1_pred, group_1_label)
                fairness_loss = torch.abs(group_0_tpr - group_1_tpr)

            case 'equalized odds':
                # Calculate both True Positive Rate (TPR) and False Positive Rate (FPR) differences
                group_0_tpr = self._get_tpr(group_0_pred, group_0_label)
                group_1_tpr = self._get_tpr(group_1_pred, group_1_label)
                group_0_fpr = self._get_fpr(group_0_pred, group_0_label)
                group_1_fpr = self._get_fpr(group_1_pred, group_1_label)
                fairness_loss = (torch.abs(group_0_tpr - group_1_tpr) + torch.abs(group_0_fpr - group_1_fpr))
            case _:
                raise NotImplementedError(f"Fairness criteria '{self.fairness_criteria}' is not supported.")
            
        return fairness_loss.mean()