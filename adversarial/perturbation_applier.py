import torch

class PerturbationApplier:
    def __init__(self, perturbation=None, perturbation_path=None, device='cpu'):
        """
        Initialize the PerturbationApplier with a given perturbation or load from a file.

        Args:
            perturbation (torch.Tensor, optional): A pre-defined perturbation tensor.
            perturbation_path (str, optional): Path to a file containing a saved perturbation tensor.
            device (str): The device (CPU or CUDA) where the perturbation will be applied.
        """
        if perturbation is not None:
            self.perturbation = perturbation.to(device).requires_grad_(True)
        elif perturbation_path is not None:
            self.perturbation = torch.load(perturbation_path).to(device).requires_grad_(True)
        else:
            raise ValueError("Either a perturbation or perturbation_path must be provided")

        # Check if the perturbation tensor has a batch size of 1
        if self.perturbation.size(0) != 1:
            raise ValueError("Perturbation tensor must have a batch size of 1")
        # Shape check for the perturbation
        if self.perturbation.ndim != 4 or self.perturbation.size(1) not in [1, 3]:
            raise ValueError("Perturbation tensor must have shape (1, C, H, W) with C being 1 or 3")

    def apply(self, images):
        """
        Apply the perturbation to a batch of images.

        Args:
            images (torch.Tensor): A batch of images to which the perturbation will be applied.

        Returns:
            torch.Tensor: The batch of images with the perturbation applied.
        """
        # Ensure the perturbation can be expanded to match the images' shape
        if images.size(1) != self.perturbation.size(1):
            raise ValueError("The number of channels in the perturbation must match the images")

        perturbation = self.perturbation.expand_as(images)
        # Apply the perturbation and clip to maintain valid image range
        perturbed_images = images + perturbation
        perturbed_images = torch.clamp(perturbed_images, 0, 1)

        return perturbed_images

    def update(self, alpha, epsilon):
        """
        Update the perturbation based on the current gradients.

        Args:
            alpha (float): Step size for the perturbation update.
            epsilon (float): Maximum allowable perturbation size.
        """
        with torch.no_grad():
            self.perturbation += alpha * self.perturbation.grad.sign()
            self.perturbation.clamp_(-epsilon, epsilon)
            self.perturbation.grad.zero_()

    def get(self):
        """
        Get the current perturbation tensor.

        Returns:
            torch.Tensor: The current perturbation tensor.
        """
        return self.perturbation