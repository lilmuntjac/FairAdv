from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

class HAM10000Dataset(Dataset):
    """Custom PyTorch Dataset class for the HAM10000 dataset."""

    def __init__(self, csv_file, img_dir, selected_attrs=None, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.attributes = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

        # Set default for selected_attrs if None
        if selected_attrs is None:
            selected_attrs = ['diagnosis', 'sex']
        self.selected_attrs = selected_attrs

        # Dictionaries for mapping labels
        self.case_dict = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
        self.gender_dict = {'male': 0, 'unknown': 0, 'female': 1}
        # This dataset also includes age, but it has not been implemented yet.

    def __len__(self):
        return len(self.attributes.index)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the item to fetch.

        Returns:
            tuple: (image, target) where target is the labels of the image attributes.
        """
        img_name = self.attributes.iloc[index]['image_id']
        img_path = self.img_dir / (img_name + '.jpg')
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        # Get attributes and convert to labels
        dx_label = self.case_dict.get(self.attributes.iloc[index]['dx'], -1)
        sex_label = self.gender_dict.get(self.attributes.iloc[index]['sex'], 0)

        target_labels = {
            'diagnosis': dx_label,
            'sex': sex_label
        }
        target = [target_labels[attr] for attr in self.selected_attrs]
        target = np.array(target, dtype=np.int64)

        return image, target
        