import torch
from torch.nn import BCELoss, CrossEntropyLoss
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from pathlib import Path
from models.binary_model import BinaryModel
from models.multiclass_model import MulticlassModel

def setup_model_and_optimizer(config, device):
    """ Set up the model structure and optimizer based on the configuration. """
    if config['dataset']['type'] == 'binary':
        model = BinaryModel(len(config['dataset']['selected_attrs'])).to(device)
        criterion = BCELoss()
    elif config['dataset']['type'] == 'multi-class':
        class_num = config['dataset']['class_number']
        model = MulticlassModel(class_num).to(device)
        criterion = CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid dataset type: {config['dataset']['type']}")
    # Setup optimizer
    learning_rate = config['training'].get('learning_rate', 1e-3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Setup scheduler
    if 'scheduler' in config['training'] and config['training']['scheduler']:
        scheduler_type  = config['training']['scheduler']
        if scheduler_type == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer)
        elif scheduler_type == "MultiStepLR":
            scheduler = MultiStepLR(optimizer, [30, 60, 90], gamma=0.1)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    else:
        scheduler = None

    return model, criterion, optimizer, scheduler

def initialize_stats_tensors(config):
    """ Initialize empty stats tensors based on the configuration """
    num_attrs = len(config['dataset'].get('selected_attrs', []))
    if config['dataset']['type'] == 'binary':
        train_epoch_stats_count = torch.empty(0, num_attrs, 2, 4)
        val_epoch_stats_count = torch.empty(0, num_attrs, 2, 4)
    elif config['dataset']['type'] == 'multi-class':
        train_epoch_stats_count = torch.empty(0, 2, 2)
        val_epoch_stats_count = torch.empty(0, 2, 2)
    else:
        raise ValueError(f"Invalid dataset type: {config['dataset']['type']}")
    return train_epoch_stats_count, val_epoch_stats_count

def load_checkpoint_and_stats(config, model, optimizer, scheduler, device):
    """ Load the model checkpoint and stats if specified in the config. """
    passed_epoch = 0
    if 'load_path' in config['training'] and config['training']['load_path']:
        load_path = Path(config['training']['load_path'])
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint file does not exist at specified path: {load_path}")
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        passed_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {passed_epoch + 1}")
    
    train_epoch_stats_count, val_epoch_stats_count = initialize_stats_tensors(config)
    if 'load_stats' in config['training'] and config['training']['load_stats']:
        stats_path = Path(config['training']['load_stats'])
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file does not exist at specified path: {stats_path}")
        loaded_stats = torch.load(stats_path)

        if loaded_stats['epoch'] != passed_epoch:
            raise ValueError(f"Stats tensor shape mismatch: expected {passed_epoch}, got {loaded_stats['epoch']} instead")
        train_epoch_stats_count = loaded_stats['train']
        val_epoch_stats_count = loaded_stats['val']

    return passed_epoch, train_epoch_stats_count, val_epoch_stats_count