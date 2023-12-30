from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.celeba_dataset import CelebADataset

def get_transforms(input_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """ Get the standard transformations for image data. """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
    ])

def create_celeba_data_loaders(
    attr_file='/tmp2/dataset/celeba/list_attr_celeba.txt',
    partition_file='/tmp2/dataset/celeba/list_eval_partition.txt',
    img_dir='/tmp2/dataset/celeba/img_align_celeba',
    selected_attrs=None,
    batch_size=128
):
    """ Create and return DataLoaders specifically for the CelebA dataset. """
    transform = get_transforms()

    train_dataset = CelebADataset(
        attr_file=attr_file,
        partition_file=partition_file,
        img_dir=img_dir,
        partition_type=0,  # 0 for train
        selected_attrs=selected_attrs,
        transform=transform
    )

    val_dataset = CelebADataset(
        attr_file=attr_file,
        partition_file=partition_file,
        img_dir=img_dir,
        partition_type=1,  # 1 for validation
        selected_attrs=selected_attrs,
        transform=transform
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