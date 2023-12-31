import torch
import torch.nn as nn
from torchvision import models

class MulticlassModel(nn.Module):
    """A multiclass classification model based on the ResNet34 architecture."""
    def __init__(self, num_classes):
        """
        Args:
            num_classes (int): The number of classes in the dataset. This determines the size of the output layer,
                               where the model outputs a probability distribution over these classes.
        """
        super(MulticlassModel, self).__init__()
        # Load pre-trained ResNet34 using the new weights argument
        self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Replace the final fully connected layer
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)