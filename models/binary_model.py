import torch
import torch.nn as nn
from torchvision import models

class BinaryModel(nn.Module):
    """A binary classification model based on the ResNet18 architecture."""
    def __init__(self, num_attributes):
        """
        Args:
            num_attributes (int): The number of attributes in the dataset. This determines the size of the output layer,
                                  where each attribute is predicted as a binary value (0 or 1).
        """
        super(BinaryModel, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Replace the final fully connected layer
        self.model.fc = nn.Linear(512, num_attributes)

    def forward(self, x):
        x = self.model(x)

        # Apply sigmoid activation
        return torch.sigmoid(x)