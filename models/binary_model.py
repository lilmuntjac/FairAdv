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
        # Initialize the base ResNet18 model with pre-trained weights
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.features = nn.Sequential(*list(self.base.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_attributes)

    def forward(self, x, get_feature=False):
        # Extract feature maps from the 4th block
        feature = self.features(x)
        pooled_output = self.avgpool(feature)
        pooled_output = torch.flatten(pooled_output, 1)
        final_output = self.fc(pooled_output)
        final_output = torch.sigmoid(final_output)

        # Return both the final output and the feature map if requested
        if get_feature:
            return final_output, feature
        else:
            return final_output
