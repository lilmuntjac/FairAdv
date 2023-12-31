from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

class FairFaceDataset(Dataset):
    """Custom PyTorch Dataset class for the FairFace dataset."""

    def __init__(self, csv_file, root_dir, selected_attrs, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Root directory for the dataset.
            selected_attrs (list of string): List of selected attribute names.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.attributes = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Set default for selected_attrs if None
        if selected_attrs is None:
            selected_attrs = ['age', 'gender', 'race']
        self.selected_attrs = selected_attrs

        # Dictionaries for mapping labels
        self.race_dict = {'White': 0, 'Black': 1, 'Latino_Hispanic': 2, 'East Asian': 3,
                          'Southeast Asian': 4, 'Indian': 5, 'Middle Eastern': 6}
        self.gender_dict = {'Male': 0, 'Female': 1}
        self.age_dict = {'0-2': 0, '3-9': 1, '10-19': 2, '20-29': 3,
                         '30-39': 4, '40-49': 5, '50-59': 6, '60-69': 7, 'more than 70': 8}

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
            image = self.transform(image)

        # Get attributes
        race = self.attributes.iloc[index]['race']
        gender = self.attributes.iloc[index]['gender']
        age_group = self.attributes.iloc[index]['age']

        # Convert attributes to labels
        race_label = 1 if race == 'White' else 0
        age = self.age_dict.get(age_group, -1)
        age_label = 1 if age >= self.age_dict.get('30-39', -1) else 0
        gender_label = self.gender_dict.get(gender, -1)

        # Build target based on selected attributes
        target_labels = {
            'race': race_label,
            'gender': gender_label,
            'age': age_label
        }
        target = [target_labels[attr] for attr in self.selected_attrs]
        target = np.array(target, dtype=np.float32)

        return image, target