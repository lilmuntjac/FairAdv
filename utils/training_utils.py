from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, CosineAnnealingLR

from models.generic_model import GenericModel
from data.loaders.dataloader import (create_celeba_data_loaders, create_celeba_xform_data_loaders,
                                     create_fairface_data_loaders, create_fairface_xform_data_loaders,
                                     create_ham10000_data_loaders)
from training import (GenericTrainer, MFDTrainer, FairPatternTrainer, 
                      FSCLSupConTrainer, FSCLClassifierTrainer, ReWeightTrainer, FHSICTrainer, AdvTrainer)

def setup_training_components(config, device):
    """
    Sets up and returns the model, optimizer, criterion, and scheduler based on the configuration.
    
    Returns:
    - Tuple containing the model, optimizer, criterion, and scheduler.
    """
    # Model
    training_schema = config['dataset'].get('training_schema', '')
    if training_schema in ['generic', 'pattern', 'mfd', 'fscl classifier', 'reweight', 'fhsic', 'adversarial',]:
        model = GenericModel(num_outputs=config['dataset']['num_outputs']).to(device)
    elif training_schema in ['fscl supcon',]:
        model = GenericModel(num_outputs=config['dataset']['num_outputs'],
                             contrastive=True).to(device)
    else:
        raise ValueError(f"Invalid training method: {training_schema}")
    
    # Optimizer
    learning_rate = float(config['training'].get('learning_rate', 1e-3))
    if training_schema in ['fscl supcon', 'fscl classifier']:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                    momentum=0.9, weight_decay=1e-4)
    elif training_schema in ['reweight',]:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                     weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Criterion
    criterion_type = config['dataset'].get('type', 'binary')
    if training_schema == 'reweight':
        if criterion_type == 'binary':
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        elif criterion_type == 'multi-class':
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
        else:
            raise ValueError(f"Unsupported criterion type: {criterion_type}")
    else:
        if criterion_type == 'binary':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif criterion_type == 'multi-class':
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion type: {criterion_type}")

    # Scheduler
    if 'scheduler' in config['training'] and config['training']['scheduler']:
        scheduler_type  = config['training']['scheduler']
        if scheduler_type == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer)
        elif scheduler_type == "MultiStepLR":
            scheduler = MultiStepLR(optimizer, [30, 60, 90], gamma=0.1)
        elif scheduler_type == "CosineAnnealingLR":
            eta_min = learning_rate * (0.1 ** 3)
            epoch = config['training'].get('final_epoch', 10)
            scheduler = CosineAnnealingLR(optimizer, T_max=epoch, eta_min=eta_min)
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
        # Initialize stats tensors based on model type in config
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
        else:
            raise ValueError(f"Invalid configuration: dataset={dataset_name}, pattern={pattern_type}")
        
    # Get the loader function parameters
    selected_attrs = config['dataset']['selected_attrs'] + [config['dataset']['protected_attr']]
    batch_size = config['training']['batch_size']
    # Get the proper sampler
    if config['dataset'].get('balanced', False):
        sampler = 'balanced_batch_sampler'
    elif config['dataset'].get('training_schema', None) == 'reweight':
        sampler = 'seeded_sampler'
    else:
        sampler = None
    return_two_versions = True if config['dataset'].get('training_schema', None) == 'fscl supcon' else False

    # Configure the dataloader based on the training method. 
    train_loader, val_loader = loader_function(selected_attrs=selected_attrs, batch_size=batch_size, 
                                               sampler=sampler, return_two_versions=return_two_versions)
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
    elif training_schema == 'fscl supcon':
        return FSCLSupConTrainer
    elif training_schema == 'fscl classifier':
        return FSCLClassifierTrainer
    elif training_schema == 'reweight':
        return ReWeightTrainer
    elif training_schema == 'fhsic':
        return FHSICTrainer
    elif training_schema == 'adversarial':
        return AdvTrainer
    else:
        raise ValueError(f"Invalid trainer method specified in config: {training_schema}")
