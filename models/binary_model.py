import torch
import torch.nn as nn
from torchvision import models

class BinaryModel(nn.Module):
    def __init__(self, num_attributes):
        super(BinaryModel, self).__init__()
        # Load pre-trained ResNet34 using the new weights argument
        self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Replace the final fully connected layer
        self.model.fc = nn.Linear(512, num_attributes)

    def forward(self, x):
        # Pass input through ResNet34
        x = self.model(x)

        # Apply sigmoid activation
        return torch.sigmoid(x)
