from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.celeba_dataset import CelebADataset
from datasets.celeba_xform_dataset import CelebAXformDataset
from datasets.fairface_dataset import FairFaceDataset
from datasets.fairface_xform_dataset import FairFaceXformDataset
from datasets.ham10000_dataset import HAM10000Dataset

def get_transforms(input_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], augment=False):
    """ Get the standard transformations for image data. """
    transforms_list = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
    ]
    if augment:
        # Insert TrivialAugmentWide at the beginning of the transforms list
        transforms_list.insert(0, transforms.TrivialAugmentWide())
    return transforms.Compose(transforms_list)
    
def create_celeba_data_loaders(
    attr_file='/tmp2/dataset/celeba/list_attr_celeba.txt',
    partition_file='/tmp2/dataset/celeba/list_eval_partition.txt',
    img_dir='/tmp2/dataset/celeba/img_align_celeba',
    selected_attrs=None,
    batch_size=128
):
    """ Create and return DataLoaders specifically for the CelebA dataset. """
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms()

    train_dataset = CelebADataset(
        attr_file=attr_file,
        partition_file=partition_file,
        img_dir=img_dir,
        partition_type=0,  # 0 for train
        selected_attrs=selected_attrs,
        transform=train_transform
    )

    val_dataset = CelebADataset(
        attr_file=attr_file,
        partition_file=partition_file,
        img_dir=img_dir,
        partition_type=2,  # 1 for validation, 2 for test
        selected_attrs=selected_attrs,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=16, pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=16, pin_memory=True, drop_last=False
    )
    return train_loader, val_loader

def create_celeba_xform_data_loaders(
    train_csv='/tmp2/dataset/celeba_tm/celeba_tm_train.csv',
    val_csv='/tmp2/dataset/celeba_tm/celeba_tm_test.csv',
    img_dir='/tmp2/dataset/celeba/img_align_celeba',
    selected_attrs=None,
    batch_size=128
):
    """ Create and return DataLoaders specifically for the CelebA dataset. """
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms()

    train_dataset = CelebAXformDataset(
        csv_file=train_csv,
        img_dir=img_dir,
        selected_attrs=selected_attrs,
        transform=train_transform
    )

    val_dataset = CelebAXformDataset(
        csv_file=val_csv,
        img_dir=img_dir,
        selected_attrs=selected_attrs,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=16, pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=16, pin_memory=True, drop_last=False
    )
    return train_loader, val_loader

def create_fairface_data_loaders(
    train_csv='/tmp2/dataset/fairface-img-margin025-trainval/fairface_label_train.csv',
    val_csv='/tmp2/dataset/fairface-img-margin025-trainval/fairface_label_val.csv',
    root_dir='/tmp2/dataset/fairface-img-margin025-trainval',
    selected_attrs=['age', 'race'],
    batch_size=128
):
    """ Create and return DataLoaders specifically for the FairFace dataset. """
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms()

    train_dataset = FairFaceDataset(
        csv_file=train_csv,
        root_dir=root_dir,
        selected_attrs=selected_attrs,
        transform=train_transform
    )

    val_dataset = FairFaceDataset(
        csv_file=val_csv,
        root_dir=root_dir,
        selected_attrs=selected_attrs,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=16, pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=16, pin_memory=True, drop_last=False
    )
    return train_loader, val_loader

def create_fairface_xform_data_loaders(
    train_csv='/tmp2/dataset/fairface-img-margin025-trainval/fairface_label_tm_train.csv',
    val_csv='/tmp2/dataset/fairface-img-margin025-trainval/fairface_label_tm_val.csv',
    root_dir='/tmp2/dataset/fairface-img-margin025-trainval',
    selected_attrs=['age', 'race'],
    batch_size=128
):
    """ Create and return DataLoaders specifically for the FairFace dataset. """
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms()

    train_dataset = FairFaceXformDataset(
        csv_file=train_csv,
        root_dir=root_dir,
        selected_attrs=selected_attrs,
        transform=train_transform
    )

    val_dataset = FairFaceXformDataset(
        csv_file=val_csv,
        root_dir=root_dir,
        selected_attrs=selected_attrs,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=16, pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=16, pin_memory=True, drop_last=False
    )
    return train_loader, val_loader

def create_ham10000_data_loaders(
        train_csv='/tmp2/dataset/HAM10000/ham10000_train.csv',
        val_csv='/tmp2/dataset/HAM10000/ham10000_val.csv',
        img_dir='/tmp2/dataset/HAM10000/train', # we don't use test set cause a lot of gender is missing
        batch_size=128
):
    """ Create and return DataLoaders specifically for the HAM10000 dataset. """
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms()

    train_dataset = HAM10000Dataset(
        csv_file=train_csv,
        img_dir=img_dir,
        transform=train_transform
    )

    val_dataset = HAM10000Dataset(
        csv_file=val_csv,
        img_dir=img_dir,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=16, pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=16, pin_memory=True, drop_last=False
    )
    return train_loader, val_loader