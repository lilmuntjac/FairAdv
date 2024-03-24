import torch
import torch.nn as nn
from torchvision import models

class MulticlassModel(nn.Module):
    """A multiclass classification model based on the ResNet18 architecture."""
    def __init__(self, num_classes):
        """
        Args:
            num_classes (int): The number of classes in the dataset. This determines the size of the output layer,
                               where the model outputs a probability distribution over these classes.
        """
        super(MulticlassModel, self).__init__()
        # Initialize the base ResNet18 model with pre-trained weights
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.features = nn.Sequential(*list(self.base.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, get_feature=False):
        # Extract feature maps from the 4th block
        feature = self.features(x)
        pooled_output = self.avgpool(feature)
        pooled_output = torch.flatten(pooled_output, 1)
        final_output = self.fc(pooled_output)

        # Return both the final output and the feature map if requested
        if get_feature:
            return final_output, feature
        else:
            return final_output