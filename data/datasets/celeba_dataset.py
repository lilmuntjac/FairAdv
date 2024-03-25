from pathlib import Path
from PIL import Image
import pandas as pd

from torch.utils.data import Dataset

class CelebADataset(Dataset):
    """Custom Dataset for loading CelebA dataset images and selected attributes,
    and optionally returning subgroup info."""

    def __init__(self, attr_file, partition_file, img_dir, partition_type, 
                 selected_attrs=None, transform=None, return_subgroups=False):
        """
        Args:
            attr_file (string): Path to the file with annotations (attributes).
            partition_file (string): Path to the file with train/val/test partitions.
            img_dir (string): Directory with all the images.
            partition_type (int): Type of dataset partition (0: Train, 1: Val, 2: Test).
            selected_attrs (list of str): List of attribute names to be included.
            transform (callable, optional): Optional transform to be applied on a sample.
            return_subgroups (bool, optional): If True, returns subgroup information
                                               along with images and attributes.
        """
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.return_subgroups = return_subgroups

        # Read the attributes file
        with open(attr_file, 'r') as f:
            lines = f.readlines()
        attr_names = lines[1].split()
        data = [line.split() for line in lines[2:]]
        attr_data = pd.DataFrame(data, columns=['filename'] + attr_names)
        attr_data[attr_names] = attr_data[attr_names].apply(pd.to_numeric)  # Convert attributes to numeric

        # Read partitions
        partition_data = pd.read_csv(partition_file, sep='\s+', header=None, names=['filename', 'partition'])
        
        # Merge attributes with partitions
        self.data = attr_data.merge(partition_data, on='filename')
        self.data = self.data[self.data['partition'] == partition_type]

        # Filter and reorder attributes if a list is provided
        if selected_attrs is not None:
            self.selected_attrs = selected_attrs
            self.data = self.data[['filename', 'partition'] + selected_attrs]
        else:
            self.selected_attrs = attr_names

        # Compute and add subgroup information based on selected attributes
        if return_subgroups:
            self.data['subgroup'] = self.data.apply(
                lambda row: self.compute_subgroup(row), axis=1)

    def compute_subgroup(self, row):
        """Computes the subgroup for a given row based on selected attributes."""
        # ((row[attr] + 1) // 2) normalizes the attribute values from -1 and 1 to 0 and 1
        return tuple(((row[attr] + 1) // 2) for attr in self.selected_attrs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['filename']
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Extract and map selected attributes
        attributes = self.data.iloc[idx][self.selected_attrs].to_numpy()
        attributes = (attributes.astype('int') + 1) // 2
        attributes = attributes.astype('float32')

        if self.return_subgroups:
            subgroup = self.data.iloc[idx]['subgroup']
            return image, attributes, subgroup
        else:
            return image, attributes


# import matplotlib.pyplot as plt
# from torchvision import transforms, utils

# # Initialize the dataset
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
# ])

# selected_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes']  # Specify the attributes of interest
# celeba_dataset = CelebADataset(attr_file='/tmp2/dataset/celeba/list_attr_celeba.txt',
#                                partition_file='/tmp2/dataset/celeba/list_eval_partition.txt',
#                                img_dir='/tmp2/dataset/celeba/img_align_celeba',
#                                partition_type=0,  # 0 for train, 1 for val, 2 for test
#                                selected_attrs=selected_attrs,
#                                transform=transform)

# # Function to save an image
# def save_image(image, attr_str, idx, output_dir='.'):
#     filename = f'image_{idx}_{attr_str.replace(" ", "_")}.png'
#     utils.save_image(image, Path(output_dir) / filename)

# # Iterate over the first three items in the dataset and save the images
# for i in range(3):
#     image, attributes = celeba_dataset[i]
#     attr_values = zip(selected_attrs, attributes)
#     attr_str = ', '.join([f'{attr}: {"Yes" if value > 0 else "No"}' for attr, value in attr_values])

#     save_image(image, attr_str, i)