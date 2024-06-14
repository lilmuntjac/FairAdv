from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from .binary_base import BinaryCrossEntropy
from .constraint import BinaryFairnessConstraint
from .equimask import BinaryEquimask

class CombinedBinaryLoss(nn.Module):
    def __init__(self, 
                 fairness_criteria, 
                 main_loss, 
                 secondary_loss=None,
                 gamma_adjustment='constant',
                 gamma=1.0,
                 gamma_adjust_factor=0.01,
                 accuracy_goal=0.8,
                 window_size=10):
        super(CombinedBinaryLoss, self).__init__()
        self.fairness_criteria = fairness_criteria
        self.gamma = gamma
        self.gamma_adjustment = gamma_adjustment
        self.gamma_adjust_factor = gamma_adjust_factor
        self.accuracy_goal = accuracy_goal
        self.window_size = window_size
        self.accuracy_history = deque(maxlen=window_size)

        self.main_loss = self.get_loss(main_loss)
        self.secondary_loss = self.get_loss(secondary_loss) if secondary_loss else None

    def get_loss(self, loss_name):
        """
        Retrieves a loss function object based on the provided loss name.
        """
        match loss_name.lower():
            case 'binary cross entropy':
                return BinaryCrossEntropy(self.fairness_criteria)
            case 'fairness constraint':
                return BinaryFairnessConstraint(self.fairness_criteria, 'gumbel_sigmoid')
            case 'crude fairness constraint':
                return BinaryFairnessConstraint(self.fairness_criteria, 'steep_function')
            case 'perturbed fairness constraint':
                return BinaryFairnessConstraint(self.fairness_criteria, 'perturbed_optimizer')
            case 'equimask':
                return BinaryEquimask(self.fairness_criteria)
            case _:
                raise NotImplementedError()
            
    def get_accuracy(self, outputs, labels):
        # Extract the protected attribute and actual labels from the labels tensor
        protected_attribute = labels[:, -1] # in shape (N,)
        actual_labels = labels[:, :-1]      # in shape (N, 1)
        outputs = torch.sigmoid(outputs)
        predictions = (outputs > 0.5).int() # in shape (N, 1)

        accuracy = (predictions == actual_labels).float().mean()
        return accuracy
    
    def adjust_gamma_based_on_accuracy(self, outputs, labels, verbose=False):
        """
        Adjusts the gamma based on the accuracy of the model compared to an accuracy goal.
        Only adjust the gamma once the history has been filled up.
        """
        current_accuracy = self.get_accuracy(outputs, labels)
        self.accuracy_history.append(current_accuracy)  # Add current accuracy to history
        if len(self.accuracy_history) == self.window_size:
            average_accuracy = torch.tensor(list(self.accuracy_history)).mean()

            if verbose:
                print(f'Current Accuracy: {current_accuracy:.4f}')
                print(f'Average Accuracy: {average_accuracy:.4f}')

            if self.gamma_adjustment == 'dynamic':
                if average_accuracy < self.accuracy_goal:
                    self.gamma *= (1 - self.gamma_adjust_factor)
                elif average_accuracy > self.accuracy_goal:
                    self.gamma *= (1 + self.gamma_adjust_factor)

                # Ensure gamma stays within reasonable bounds
                self.gamma = min(max(self.gamma, 0.00001), 10.0)
                if verbose:
                    print(f'Gamma after adjustment: {self.gamma:.4f}')
        # No adjustment is made if 'constant' is selected

    def forward(self, outputs, labels, applier, train_mode=True):
        if self.secondary_loss:
            # Calculate losses
            main_loss = self.main_loss.compute_loss(outputs, labels)
            secondary_loss = self.secondary_loss.compute_loss(outputs, labels)

            if train_mode:
                self.adjust_gamma_based_on_accuracy(outputs, labels, verbose=False)

            combined_loss = main_loss + self.gamma * secondary_loss
            return combined_loss
        else: 
            applier.clear_gradient()
            main_loss = self.main_loss.compute_loss(outputs, labels)
            
            return main_loss
