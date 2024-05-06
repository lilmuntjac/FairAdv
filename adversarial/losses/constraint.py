import torch
import torch.nn as nn
import torch.nn.functional as F

from .binary_base import BinaryLossBase
from .perturbed_optimizer import perturbed

class BinaryFairnessConstraint(BinaryLossBase):
    def __init__(self, fairness_criteria, use_perturbed_optimizer=False):
        """
        Initializes the BinaryFairnessConstraint class.

        Args:
            fairness_criteria (str): The type of fairness criteria to enforce.
            use_perturbed_optimizer (bool): If True, uses a perturbed optimizer to enable
                                            gradient-based optimization through non-differentiable
                                            operations. If False, uses Gumbel Softmax straight-through
                                            estimator for non-differentiable operations.
        """
        super().__init__(fairness_criteria)
        self.use_perturbed_optimizer = use_perturbed_optimizer
        if not self.use_perturbed_optimizer:
            self.gumbel_softmax = GumbelSoftmax(tau=0.1, hard=True)

    def _compute_loss(self, outputs, labels):
        """
        Conpute the fairness constraint based on the specified fairness criteria.

        Args:
            outputs (torch.Tensor): The logits from a model.
            labels (torch.Tensor): Ground truth labels with protected attributes.

        Returns:
            torch.Tensor: Calculated fairness loss.
        """
        if self.use_perturbed_optimizer:
            def attach_labels_to_constraint(outputs, labels=labels):
                return self.padded_constraint(outputs, labels)
            loss_function = perturbed(attach_labels_to_constraint, num_samples=10000, sigma=0.5,
                                      noise='gumbel', batched=False, device=labels.device)
            loss = loss_function(outputs)
            return loss.mean()
        else:
            loss = self.get_approximate_constraint(outputs, labels)
            return loss.mean()
    
    def get_approximate_constraint(self, outputs, labels):
        """
        Computes the fairness constraint using Gumbel Softmax straight-through estimator.

        Args:
            outputs (torch.Tensor): Model outputs processed through Gumbel softmax.
            labels (torch.Tensor): Ground truth labels including protected attributes.

        Returns:
            torch.Tensor: Fairness loss calculated according to the specified fairness criteria.
        """
        predictions_soft = self.gumbel_softmax(outputs) # in shape (N, 2), near one-hot encoding
        # There are only 2 classes, so the positive class is the second column
        predictions_soft = predictions_soft[:, 1]

        # Split data by protected attribute
        # should be in shape (n,) and (m,), with n+m = N
        protected_attribute = labels[:, -1] # in shape (N,)
        actual_labels       = labels[:,  0] # in shape (N,)
        group_0_pred,  group_1_pred  = self._split_by_protected_attribute(predictions_soft, protected_attribute)
        group_0_label, group_1_label = self._split_by_protected_attribute(actual_labels, protected_attribute)

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

    def padded_constraint(self, outputs, labels):
        """
        Adjusts labels to match the perturbed outputs and computes the fairness constraint.
        
        Args:
            outputs (torch.Tensor): Perturbed outputs from the model.
            labels (torch.Tensor): Ground truth labels, including protected attributes.

        Returns:
            torch.Tensor: Fairness loss based on adjusted labels and perturbed outputs.
        """
        # Forced convert outputs into predictions
        _, predictions = torch.max(outputs, 2) # 0: Perturbed dim, 1: Batch dim, 2: Subgroups dim

        # padded label into the same shape as outputs
        padded_actual_labels = labels[:, 0].repeat(outputs.shape[0], 1)
        protected_attribute = labels[:, -1]

        # Enable the constraint to be computed alone with the padded outputs
        # Split data by protected attribute
        group_0_pred,  group_1_pred  = self._split_by_protected_attribute(predictions, protected_attribute, split_dim=1)
        group_0_label, group_1_label = self._split_by_protected_attribute(padded_actual_labels, protected_attribute, split_dim=1)

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

class GumbelSoftmax(nn.Module):
    def __init__(self, tau=1.0, hard=False):
        super(GumbelSoftmax, self).__init__()
        self.tau = tau
        self.hard = hard

    def forward(self, logits):
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / self.tau
        y_soft = F.softmax(gumbels, dim=1)

        if self.hard:
            # Straight-through version
            index = y_soft.max(dim=1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(1, index, 1.0)
            return y_hard - y_soft.detach() + y_soft
        else:
            return y_soft
        