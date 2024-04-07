import torch
import torch.nn.functional as F

from .base_binary import BinaryBaseLoss
from .perturbed_optimizer import perturbed

class BinaryPerturbedLoss(BinaryBaseLoss):
    def __init__(self, fairness_criteria):
        super().__init__(fairness_criteria)

    def _compute_fairness_loss(self, outputs, labels):
        """
        Computes the pertrubed fairness loss based on the specified fairness criteria.

        Args:
            outputs (torch.Tensor): Model predictions.
            labels (torch.Tensor): Ground truth labels with protected attributes.

        Returns:
            torch.Tensor: Calculated fairness loss.
        """
        # Create a corresponding loss function that 
        # can be transformed into a differentiable one by a perturbed optimizer.

        # wrapper function that set the labels first
        def p_dim_loss_label_wrapper(outputs, labels=labels):
            return self._p_dimension_fairness_loss_stem(outputs, labels)

        fairness_loss_function = perturbed(p_dim_loss_label_wrapper, num_samples=10000, sigma=0.5,
                                           noise='gumbel', batched=False, device=labels.device)
        fairness_loss = fairness_loss_function(outputs)
        return fairness_loss.mean()

    def _p_dimension_fairness_loss_stem(self, outputs, labels):
        # Convert raw outputs into predictions
        predictions = torch.where(outputs > 0.5, 1, 0)

        # Labels are extended to include the p dimension and exclude the protected attribute
        p_dim_lables = labels[:, :-1].repeat(outputs.shape[0], 1, 1)
        protected_attribute = labels[:, -1]

        # Split data by protected attribute
        group_0_pred, group_1_pred = self._split_by_protected_attribute(predictions, protected_attribute, split_dim=1)
        group_0_label, group_1_label = self._split_by_protected_attribute(p_dim_lables, protected_attribute, split_dim=1)

        # Calculate fairness loss based on specified criteria
        match self.fairness_criteria:
            case 'equality of opportunity':
                # Calculate the True Positive Rate (TPR) difference between groups
                group_0_tpr = self._get_tpr(group_0_pred, group_0_label, split_dim=1)
                group_1_tpr = self._get_tpr(group_1_pred, group_1_label, split_dim=1)
                fairness_loss = torch.abs(group_0_tpr - group_1_tpr)

            case 'equalized odds':
                # Calculate both True Positive Rate (TPR) and False Positive Rate (FPR) differences
                group_0_tpr = self._get_tpr(group_0_pred, group_0_label, split_dim=1)
                group_1_tpr = self._get_tpr(group_1_pred, group_1_label, split_dim=1)
                group_0_fpr = self._get_fpr(group_0_pred, group_0_label, split_dim=1)
                group_1_fpr = self._get_fpr(group_1_pred, group_1_label, split_dim=1)
                fairness_loss = (torch.abs(group_0_tpr - group_1_tpr) + torch.abs(group_0_fpr - group_1_fpr))
            case _:
                raise NotImplementedError(f"Fairness criteria '{self.fairness_criteria}' is not supported.")

        return fairness_loss

