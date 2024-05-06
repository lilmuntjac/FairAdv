import torch

class MulticlassLossBase:
    def __init__(self, fairness_criteria):
        # Can be "equalized precision"
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
        return self._compute_loss(outputs, labels)
    
    def _compute_loss(self, outputs, labels):
        raise NotImplementedError("Subclasses must implement this method.")