from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class EyeglassesApplier:
    def __init__(self, eyeglasses=None, eyeglasses_path=None,
                 mask_path='./adversarial/masks/eyeglasses_mask_6percent.png', device='cpu'):
        """
        Initialize the EyeglassesApplier with a given eyeglasses or load from a file.

        Args:
            eyeglasses (torch.Tensor, optional): A pre-defined eyeglasses tensor.
            eyeglasses_path (str, optional): Path to a file containing a saved eyeglasses tensor.
            mask_path (str): Path to an image of eyeglasses mask.
            device (str): The device (CPU or CUDA) where the eyeglasses will be applied.
        """
        if eyeglasses is not None:
            self.eyeglasses = eyeglasses.to(device).requires_grad_(True)
        elif eyeglasses_path is not None:
            self.eyeglasses = torch.load(eyeglasses_path).to(device).requires_grad_(True)
        else:
            raise ValueError("Either a eyeglasses or eyeglasses_path must be provided")
        
        # Load mask
        mask_image = Image.open(mask_path)
        self.mask = TF.to_tensor(mask_image).unsqueeze(0).to(device)

        # Check if the eyeglasses tensor has a batch size of 1
        if self.eyeglasses.size(0) != 1:
            raise ValueError("Eyeglasses tensor must have a batch size of 1")
        # Shape check for the eyeglasses
        if self.eyeglasses.ndim != 4 or self.eyeglasses.size(1) not in [1, 3]:
            raise ValueError("Eyeglasses tensor must have shape (1, C, H, W) with C being 1 or 3")
        
    def apply(self, images, theta):
        """
        Apply the eyeglasses to a batch of images.

        Args:
            images (torch.Tensor): A batch of images to which the eyeglasses will be applied.
            theta (torch.Tensor): A batch of affine matrices with shape (Nx2x3)
        Returns:
            torch.Tensor: The batch of images with the eyeglasses applied.
        """
        # Ensure the eyeglasses can be expanded to match the images' shape
        if images.size(1) != self.eyeglasses.size(1):
            raise ValueError("The number of channels in the eyeglasses must match the images")
        # Ensure theta has the correct shape
        if theta.shape[1:] != (2, 3):
            raise ValueError("Theta must have a shape of (N, 2, 3)")

        
        mask = self.mask.expand_as(images)
        eyeglasses = self.eyeglasses.expand_as(images)
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        xform_mask = F.grid_sample(mask, grid, align_corners=False)
        xform_eyeglasses = F.grid_sample(eyeglasses, grid, mode='bilinear', align_corners=False)

        # Apply the eyeglasses
        eyeglasses_applied_images = images * (1 - xform_mask) + xform_eyeglasses * xform_mask
        
        return eyeglasses_applied_images

    def update(self, alpha):
        """
        Update the eyeglasses based on the current gradients.

        Args:
            alpha (float): Step size for the eyeglasses update.
        """
        with torch.no_grad():
            self.eyeglasses += alpha * self.eyeglasses.grad.sign()
            self.eyeglasses.clamp_(0, 1)
            self.eyeglasses.grad.zero_()


    def get(self):
        """
        Get the current eyeglasses tensor.

        Returns:
            torch.Tensor: The current eyeglasses tensor.
        """
        return self.eyeglasses