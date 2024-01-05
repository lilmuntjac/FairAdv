from pathlib import Path
from PIL import Image
import pandas as pd

from torch.utils.data import Dataset

class CelebAXformDataset(Dataset):
    """Custom Dataset for loading CelebA dataset with affine transformation matrices."""

    def __init__(self, csv_file, img_dir, selected_attrs=None, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            img_dir (string): Directory with all the images.
            selected_attrs (list of string): List of selected attribute names.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.attributes_full = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

        # Filter and reorder columns based on selected attributes
        mandatory_cols = ['filename']  # Mandatory column
        theta_cols = ['a11', 'a12', 'a13', 'a21', 'a22', 'a23']  # Columns for affine matrix
        # If selected attributes are not provided, use all attributes from the CSV
        if selected_attrs is None:
            # Exclude filename and theta columns to get only attribute columns
            all_attr_cols = [col for col in self.attributes_full.columns if col not in mandatory_cols + theta_cols]
            selected_attrs = all_attr_cols

        # Combine all columns needed
        all_cols = mandatory_cols + selected_attrs + theta_cols
        self.attributes = self.attributes_full[all_cols]

    def __len__(self):
        return len(self.attributes.index)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the item to fetch.

        Returns:
            tuple: (image, theta, target) where theta is the affine transformation matrix and
                   target is a tensor of selected attributes.
        """
        img_name = self.attributes.iloc[idx]['filename']
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Extract theta values as a numpy array
        theta_values = self.attributes.iloc[idx, -6:].values.astype('float32')
        theta = theta_values.reshape(2, 3)
        # Extract target attributes
        target_attrs = self.attributes.iloc[idx, 1:-6].values.astype('float32')
        target = target_attrs

        return image, theta, target

        


        