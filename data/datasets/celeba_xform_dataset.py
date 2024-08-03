from pathlib import Path
from PIL import Image
import pandas as pd

from torch.utils.data import Dataset

class CelebAXformDataset(Dataset):
    """Custom Dataset for loading CelebA dataset images with affine transformation matrices
    and optionally returning subgroup information."""

    def __init__(self, csv_file, img_dir, selected_attrs=None, transform=None, 
                 return_subgroups=False, return_two_versions=False):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            img_dir (string): Directory with all the images.
            selected_attrs (list of string): List of selected attribute names.
            transform (callable, optional): Optional transform to be applied on a sample.
            return_subgroups (bool, optional): If True, includes subgroup information
                                               with each returned item.
            return_two_versions (bool, optional): If True, returns two augmented versions 
                                                  of each image.
        """
        self.attributes_full = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.return_subgroups = return_subgroups
        self.return_two_versions = return_two_versions

        # Filter and reorder columns based on selected attributes
        mandatory_cols = ['filename']  # Mandatory column
        self.theta_cols = ['a11', 'a12', 'a13', 'a21', 'a22', 'a23']  # Columns for affine matrix


        # If selected attributes are not provided, use all attributes from the CSV
        if selected_attrs is None:
            # Exclude filename and theta columns to get only attribute columns
            all_attr_cols = [col for col in self.attributes_full.columns 
                             if col not in mandatory_cols + self.theta_cols]
            self.selected_attrs = all_attr_cols
        else:
            self.selected_attrs = [col for col in selected_attrs 
                                   if col not in mandatory_cols + self.theta_cols]

        # Combine all columns needed
        all_cols = mandatory_cols + self.selected_attrs + self.theta_cols
        self.data = self.attributes_full.loc[:, all_cols]

        # Compute and add subgroup information based on selected attributes
        if return_subgroups:
            self.data['subgroup'] = self.data.apply(
                lambda row: self.compute_subgroup(row), axis=1)

    def compute_subgroup(self, row):
        """Computes the subgroup for a given row based on selected attributes."""
        # ((row[attr] + 1) // 2) normalizes the attribute values from -1 and 1 to 0 and 1
        return tuple(((row[attr] + 1) // 2) for attr in self.selected_attrs)

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the item to fetch.

        Returns:
            tuple: (image, theta, target) where theta is the affine transformation matrix and
                   target is a tensor of selected attributes.
        """
        img_name = self.data.iloc[idx]['filename']
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            if self.return_two_versions:
                image1 = self.transform(image)
                image2 = self.transform(image)
            else:
                image = self.transform(image)

        # Extract theta values as a numpy array
        theta_values = self.data.iloc[idx][self.theta_cols].values.astype('float32')
        theta = theta_values.reshape(2, 3)
        # Extract target attributes
        target_attrs = self.data.iloc[idx][self.selected_attrs].values.astype('int')
        target_attrs = (target_attrs + 1) // 2
        target = target_attrs

        if self.return_subgroups:
            subgroup = self.data.iloc[idx]['subgroup']
            if self.return_two_versions:
                return (image1, image2), theta, target, subgroup
            else:
                return image, theta, target, subgroup
        else:
            if self.return_two_versions:
                return (image1, image2), theta, target
            else:
                return image, theta, target
