from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

from models.generic_model import GenericModel
from data.loaders.dataloader import (create_celeba_data_loaders, create_celeba_xform_data_loaders,
                                     create_fairface_data_loaders, create_fairface_xform_data_loaders,
                                     create_ham10000_data_loaders)
from training import GenericTrainer, MFDTrainer, FairPatternTrainer

def setup_training_components(config, device):
    """
    Sets up and returns the model, optimizer, criterion, and scheduler based on the configuration.
    
    Returns:
    - Tuple containing the model, optimizer, criterion, and scheduler.
    """
    # Model
    training_schema = config['dataset'].get('training_schema', '')
    if training_schema in ['generic', 'pattern', 'mfd']:
        model = GenericModel(num_subgroups=config['dataset']['num_subgroups']).to(device)
    elif training_schema in ['contrastive',]:
        model = GenericModel(contrastive=True).to(device)
    else:
        raise ValueError(f"Invalid training method: {training_schema}")
    
    # Optimizer
    learning_rate = float(config['training'].get('learning_rate', 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Scheduler
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
    return model, optimizer, criterion, scheduler

def load_training_components(config, model, optimizer, scheduler, device):
    """
    Loads training components from a checkpoint if specified in the configuration.
    Args:
    - config (dict): Configuration dictionary.
    - model (torch.nn.Module): Model to load state into.
    - optimizer (torch.optim.Optimizer): Optimizer to load state into.
    - scheduler (torch.optim.lr_scheduler): Scheduler to load state into.
    - device (str): Device to map the loaded states.

    Returns:
    - The epoch number to resume training from.
    """
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
        print(f"Loaded checkpoint for setup. Resuming from epoch {passed_epoch + 1}.")
    else:
        print("Initializing environment for a new training session.")
    return passed_epoch

def setup_stats_tensors(config, passed_epoch):
    """
    Loads or initializes statistics tensors based on the configuration.
    Args:
    - config (dict): Configuration dictionary.
    - passed_epoch (int): The last completed epoch, for validating loaded stats.

    Returns:
    - Tuple of tensors for training and validation statistics.
    """
    if 'load_stats' in config['training'] and config['training']['load_stats']:
        stats_path = Path(config['training']['load_stats'])
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file does not exist at specified path: {stats_path}")
        loaded_stats = torch.load(stats_path)

        if loaded_stats['epoch'] != passed_epoch:
            raise ValueError(f"Stats tensor shape mismatch: expected {passed_epoch}, got {loaded_stats['epoch']} instead")
        train_stats, val_stats = loaded_stats['train'], loaded_stats['val']
    else:
        # IInitialize stats tensors based on model type in config
        model_type = config['dataset'].get('type', 'binary')
        if model_type == 'binary':
            # For binary: 2 groups, counts of [TP, FP, FN, TN]
            train_stats, val_stats = torch.empty(0, 2, 4), torch.empty(0, 2, 4)
        elif model_type == 'multi-class':
            # For multi-class: 2 groups, counts of correct and incorrect predictions
            train_stats, val_stats = torch.empty(0, 2, 2), torch.empty(0, 2, 2)
        else:
            raise ValueError(f"Unknown model prediction type {model_type}.")

    return train_stats, val_stats

def select_data_loader(config):
    """
    Selects the appropriate data loader based on the dataset name 
    and pattern type specified in config.

    Returns:
    - A tuple of (train_loader, val_loader) configured as specified.
    """
    dataset_name = config['dataset']['name']
    pattern_type = config.get('attack', {}).get('pattern_type', 'perturbation')

    # Loader function with or without a transformation matrix.
    if pattern_type == 'eyeglasses':
        if dataset_name == 'celeba':
            loader_function = create_celeba_xform_data_loaders
        elif dataset_name == 'fairface':
            loader_function = create_fairface_xform_data_loaders
        elif dataset_name == 'ham10000':
            raise ValueError('Cannot apply eyeglasses onto HAM10000 dataset')
    else:
        # Default behavior for 'perturbation' and 'frame'
        if dataset_name == 'celeba':
            loader_function = create_celeba_data_loaders
        elif dataset_name == 'fairface':
            loader_function = create_fairface_data_loaders
        elif dataset_name == 'ham10000':
            loader_function = create_ham10000_data_loaders
            train_loader, val_loader = loader_function(batch_size=config['training']['batch_size'])
            return train_loader, val_loader
        else:
            raise ValueError(f"Invalid configuration: dataset={dataset_name}, pattern={pattern_type}")
        
    balanced = config['dataset'].get('balanced', False)
    # Configure the dataloader based on the training method. 
    if not balanced:
        train_loader, val_loader = loader_function(
            selected_attrs=config['dataset']['selected_attrs'] + [config['dataset']['protected_attr']],
            batch_size=config['training']['batch_size']
        )
    else:
        train_loader, val_loader = loader_function(
            selected_attrs=config['dataset']['selected_attrs'] + [config['dataset']['protected_attr']],
            batch_size=config['training']['batch_size'],
            sampler='balanced_batch_sampler' # BalancedBatchSampler
        )
    return train_loader, val_loader

def select_trainer(config):
    """
    Selects the appropriate trainer class based on the training schema specified in config.

    Returns:
    - Trainer class corresponding to the specified schema.
    """
    training_schema = config['dataset'].get('training_schema')
    if training_schema == 'generic':
        return GenericTrainer
    elif training_schema == 'pattern':
        return FairPatternTrainer
    elif training_schema =='mfd':
        return MFDTrainer
    else:
        raise ValueError(f"Invalid trainer method specified in config: {training_schema}")
