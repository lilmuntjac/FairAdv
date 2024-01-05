import torch

class FrameApplier:
    def __init__(self, frame_thickness, frame=None, frame_path=None, device='cpu'):
        """
        Initialize the FrameApplier with a given frame or load from a file.

        Args:
            frame_thickness (float): Thickness of the frame compared to the image width.
            frame (torch.Tensor, optional): A pre-defined frame tensor.
            frame_path (str, optional): Path to a file containing a saved frame tensor.
            device (str): The device (CPU or CUDA) where the frame will be applied.
        """
        self.frame_thickness = frame_thickness
        if frame is not None:
            self.frame = frame.to(device).requires_grad_(True)
        elif frame_path is not None:
            self.frame = torch.load(frame_path).to(device).requires_grad_(True)
        else:
            raise ValueError("Either a frame or frame_path must be provided")
        
        # Check if the frame tensor has a batch size of 1
        if self.frame.size(0) != 1:
            raise ValueError("Frame tensor must have a batch size of 1")
        # Shape check for the frame
        if self.frame.ndim != 4 or self.frame.size(1) not in [1, 3]:
            raise ValueError("Frame tensor must have shape (1, C, H, W) with C being 1 or 3")
        
    def apply(self, images):
        """
        Apply the frame to a batch of images.

        Args:
            images (torch.Tensor): A batch of images to which the frame will be applied.

        Returns:
            torch.Tensor: The batch of images with the frame applied.
        """
        # Ensure the frame can be expanded to match the images' shape
        if images.size(1) != self.frame.size(1):
            raise ValueError("The number of channels in the frame must match the images")
        
        # Calculate frame thickness in pixels
        thickness = int(self.frame_thickness * images.size(3))  # Assume square images
        # Create a mask for the frame
        mask = torch.ones_like(images)
        mask[:, :, thickness:-thickness, thickness:-thickness] = 0
        # Apply the frame
        frame = self.frame.expand_as(images)
        frame_applied_images = images * (1 - mask) + frame * mask

        return frame_applied_images

    def update(self, alpha):
        """
        Update the frame based on the current gradients.

        Args:
            alpha (float): Step size for the frame update.
        """
        with torch.no_grad():
            self.frame += alpha * self.frame.grad.sign()
            self.frame.clamp_(0, 1)
            self.frame.grad.zero_()


    def get(self):
        """
        Get the current frame tensor.

        Returns:
            torch.Tensor: The current frame tensor.
        """
        return self.frame