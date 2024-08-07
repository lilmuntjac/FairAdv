from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

class FairFaceDataset(Dataset):
    """Custom PyTorch Dataset class for the FairFace dataset."""

    def __init__(self, csv_file, root_dir, selected_attrs=None, transform=None,
                 return_subgroups=False, return_two_versions=False):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Root directory for the dataset.
            selected_attrs (list of string): List of selected attribute names.
            transform (callable, optional): Optional transform to be applied on a sample.
            return_subgroups (bool, optional): If True, returns subgroup information
                                               along with images and attributes.
            return_two_versions (bool, optional): If True, returns two augmented versions of each image.
        """
        self.attributes = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.return_subgroups = return_subgroups
        self.return_two_versions = return_two_versions
        
        # Set default for selected_attrs if None
        if selected_attrs is None:
            selected_attrs = ['age', 'gender', 'race']
        self.selected_attrs = selected_attrs

        # Dictionaries for mapping labels
        # self.race_dict = {'White': 0, 'Black': 1, 'Latino_Hispanic': 2, 'East Asian': 3,
        #                   'Southeast Asian': 4, 'Indian': 5, 'Middle Eastern': 6}
        self.gender_dict = {'Male': 0, 'Female': 1}
        # self.age_dict = {'0-2': 0, '3-9': 1, '10-19': 2, '20-29': 3,
        #                  '30-39': 4, '40-49': 5, '50-59': 6, '60-69': 7, 'more than 70': 8}
        
        self.race_dict = {'White': 1, 'Black': 0, 'Latino_Hispanic': 0, 'East Asian': 0,
                          'Southeast Asian': 0, 'Indian': 0, 'Middle Eastern': 0}
        self.age_dict = {'0-2': 0, '3-9': 0, '10-19': 0, '20-29': 0,
                         '30-39': 1, '40-49': 1, '50-59': 1, '60-69': 1, 'more than 70': 1}
        
        # Compute and add subgroup information based on selected attributes
        if return_subgroups:
            self.attributes['subgroup'] = self.attributes.apply(
                lambda row: self.compute_subgroup(row), axis=1)
        
    def compute_subgroup(self, row):
        """Computes the subgroup for a given row based on selected attributes."""
        subgroup = []
        for attr in self.selected_attrs:
            if attr == 'race':
                subgroup.append(self.race_dict.get(row[attr], -1))
            elif attr == 'gender':
                subgroup.append(self.gender_dict.get(row[attr], -1))
            elif attr == 'age':
                subgroup.append(self.age_dict.get(row[attr], -1))
        return tuple(subgroup)
        
    def __len__(self):
        return len(self.attributes.index)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the item to fetch.

        Returns:
            tuple: (image, target) where target is the labels of the image attributes.
        """
        img_name = self.attributes.iloc[index]['file']
        img_path = self.root_dir / img_name
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            if self.return_two_versions:
                image1 = self.transform(image)
                image2 = self.transform(image)
            else:
                image = self.transform(image)

        # Extract and convert selected attributes to labels
        target = []
        for attr in self.selected_attrs:
            if attr == 'race':
                target.append(self.race_dict.get(self.attributes.iloc[index][attr], -1))
            elif attr == 'gender':
                target.append(self.gender_dict.get(self.attributes.iloc[index][attr], -1))
            elif attr == 'age':
                target.append(self.age_dict.get(self.attributes.iloc[index][attr], -1))
        target = np.array(target, dtype=np.float32)

        if self.return_subgroups:
            subgroup = self.attributes.iloc[index]['subgroup']
            if self.return_two_versions:
                return (image1, image2), target, subgroup
            else:
                return image, target, subgroup
        else:
            if self.return_two_versions:
                return (image1, image2), target
            else:
                return image, target
