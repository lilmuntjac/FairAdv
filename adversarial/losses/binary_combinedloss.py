import torch
import torch.nn as nn

from .binary_base import CrossEntropy
from .constraint import BinaryFairnessConstraint
from .equimask import BinaryEquimask

class CombinedBinaryLoss(nn.Module):
    def __init__(self, 
                 fairness_criteria, 
                 main_loss, 
                 secondary_loss=None,
                 gamma_adjustment='constant',
                 gamma=1.0,):
        super(CombinedBinaryLoss, self).__init__()
        self.fairness_criteria = fairness_criteria
        self.gamma = gamma
        self.gamma_adjustment = gamma_adjustment

        self.main_loss = self.get_loss(main_loss)
        self.secondary_loss = self.get_loss(secondary_loss) if secondary_loss else None

    def get_loss(self, loss_name):
        """
        Retrieves a loss function object based on the provided loss name.
        """
        match loss_name.lower():
            case 'cross entropy':
                return CrossEntropy(self.fairness_criteria)
            case 'fairness constraint':
                return BinaryFairnessConstraint(self.fairness_criteria, use_perturbed_optimizer=False)
            case 'perturbed fairness constraint':
                return BinaryFairnessConstraint(self.fairness_criteria, use_perturbed_optimizer=True)
            case 'equimask':
                return BinaryEquimask(self.fairness_criteria)
            case _:
                raise NotImplementedError()
            
    def adjust_gamma_based_on_gradient(self, main_grad, secondary_grad):
        """
        Adjusts the gamma based on the gradient magnitudes of 2 losses.

        Args:
            main_grad (torch.Tensor): Gradient of the main loss.
            secondary_grad (torch.Tensor): Gradient of the secondary loss.
        """
        if self.gamma_adjustment == 'dynamic':
            main_norm = main_grad.norm()
            secondary_norm = secondary_grad.norm()

            ratio = secondary_norm / main_norm
            desired_gamma = self.gamma / ratio.item()
            self.gamma += 0.1 * (desired_gamma - self.gamma)
            self.gamma = max(min(self.gamma, 10), 0.001)

            print(f"Main Gradient Norm: {main_norm}")
            print(f"Secondary Gradient Norm: {secondary_norm}")
            print(f"Current Gamma: {self.gamma}")
            print(f"Gradient Ratio (secondary/main): {ratio}")
            print(f"Adjusted Gamma: {self.gamma}")
        # No adjustment is made if 'constant' is selected

    def forward(self, outputs, labels, applier, train_mode=True):
        if self.secondary_loss:
            # Calculate losses
            main_loss = self.main_loss.compute_loss(outputs, labels)
            secondary_loss = self.secondary_loss.compute_loss(outputs, labels)

            if train_mode:
                # Get the gradients from both losses
                applier.clear_gradient()
                main_loss.backward(retain_graph=True)
                main_loss_grad = applier.get_gradient().clone()
                applier.clear_gradient()
                secondary_loss.backward(retain_graph=True)
                secondary_loss_grad = applier.get_gradient().clone()

                # Adjust gamma based on the gradient magnitudes
                self.adjust_gamma_based_on_gradient(main_loss_grad, secondary_loss_grad)
                applier.clear_gradient()

            combined_loss = main_loss + self.gamma * secondary_loss
            return combined_loss
        else: 
            applier.clear_gradient()
            main_loss = self.main_loss.compute_loss(outputs, labels)
            
            return main_loss
