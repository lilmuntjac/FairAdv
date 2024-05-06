import torch
import  torch.nn as nn
from torchvision import models

class GenericModel(nn.Module):
    """model that based on ResNet18 architecture"""
    def __init__(self, num_subgroups=2, contrastive=False):
        """
        Args:
            num_subgroups (int): The number of subgroups (or classes) for the classification task.
            contrastive (bool): A flag indicating whether the model should be initialized in contrastive mode,
                                which determines the inclusion of a contrastive projection head
        """
        super(GenericModel, self).__init__()

        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(self.base.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # projection head for contrastive learning
        if contrastive:
            self.contrastive_head  = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 128)
            )
        else:
            self.contrastive_head = None
        # classification head for generic training
        self.classification_head = nn.Linear(512, num_subgroups)

    def forward(self, x, get_feature=False, contrastive=False):
        """
        Forward pass through the model. Based on the mode specified, it uses either the contrastive head or
        the classification head.

        Args:
            x (torch.Tensor): Input images.
            get_feature (bool): A flag indicating whether to return the feature maps along with the final output.
            contrastive (bool): A flag indicating whether the model should operate in contrastive mode.

        Returns:
            torch.Tensor: The output logits from the appropriate head. If `get_feature` is True, a tuple of
                          (logits, feature maps) is returned.
        """
        # Extract feature maps from the 4th block
        feature = self.features(x)
        pooled_output = self.avgpool(feature)
        pooled_output = torch.flatten(pooled_output, 1)

        # Decide which head to use
        if contrastive and self.contrastive_head:
            final_output = self.contrastive_head(pooled_output)
        else:
            final_output = self.classification_head(pooled_output)

        # Return both the final output and the feature map if requested
        if get_feature:
            return final_output, feature
        else:
            return final_output