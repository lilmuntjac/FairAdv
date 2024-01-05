import os
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import utils.utils as utils
from adversarial.losses.direct_loss import binary_eqodd_loss
from models.binary_model import BinaryModel
from adversarial.perturbation_applier import PerturbationApplier
from utils.data_loader import (create_celeba_data_loaders, 
                               create_fairface_data_loaders)

def attack(model, images, labels, applier, epsilon, alpha, iters):
    for _ in range(iters):
        perturbed_images = applier.apply(images)
        perturbed_images = utils.normalize(perturbed_images)
        outputs = model(perturbed_images)
        fair_loss = binary_eqodd_loss(outputs, labels)
        # fair loss only
        loss = torch.mean(fair_loss)
        # loss = F.binary_cross_entropy(outputs, labels[:, :-1])
        loss.backward()
        applier.update(alpha, epsilon)

    predicted = (outputs > 0.5).float()
    conf_matrix = utils.get_confusion_matrix_counts(predicted, labels)

    return conf_matrix

def validation(model, images, labels, applier):
    perturbed_images = applier.apply(images)
    perturbed_images = utils.normalize(perturbed_images)
    outputs = model(perturbed_images)
    predicted = (outputs > 0.5).float()
    conf_matrix = utils.get_confusion_matrix_counts(predicted, labels)

    return conf_matrix

def run_adversarial(model, train_loader, val_loader, attack_params, device, save_path, attr_list, base_path=None):
    """Run adversarial attack and evaluate on train and validation datasets."""
    train_epoch_conf_matrices = []
    val_epoch_conf_matrices = []
    model.eval()
    # Initialize applier
    if base_path:
        applier = PerturbationApplier(perturbation_path=Path(base_path), device=device)
    else:
        # Create a random perturbation tensor if no path is provided
        random_perturbation = (torch.rand_like(next(iter(train_loader))[0][0]) * 2 - 1) * attack_params['epsilon']
        random_perturbation = random_perturbation.to(device).unsqueeze(0)  # Add batch dimension
        applier = PerturbationApplier(perturbation=random_perturbation, device=device)

    # Adversarial training loop
    for epoch in range(attack_params['epoch']):
        # Training phase
        train_conf_matrices = None
        for i, (images, labels) in enumerate(train_loader):
            if i > 100: # early stop for debug
                break
            images, labels = images.to(device), labels.to(device)
            conf_matrix = attack(model, images, labels, applier, 
                                 attack_params['epsilon'], attack_params['alpha'], 
                                 attack_params['iters'])
            if train_conf_matrices is None:
                train_conf_matrices = conf_matrix
            else:
                train_conf_matrices += conf_matrix
        train_epoch_conf_matrices.append(train_conf_matrices)
    
        # Validation phase
        val_conf_matrices = None
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                conf_matrix = validation(model, images, labels, applier)
                if val_conf_matrices is None:
                    val_conf_matrices = conf_matrix
                else:
                    val_conf_matrices += conf_matrix
            val_epoch_conf_matrices.append(val_conf_matrices)

        print(f'\nEpoch {epoch + 1} Summary:')

        # Compute and print metrics for each attribute
        for attr_index, attr_name in enumerate(attr_list):
            # Extract confusion matrices for the current attribute and epoch
            train_group1_matrix = train_conf_matrices[attr_index, 0]
            train_group2_matrix = train_conf_matrices[attr_index, 1]
            val_group1_matrix = val_conf_matrices[attr_index, 0]
            val_group2_matrix = val_conf_matrices[attr_index, 1]

            # Calculate metrics for the current attribute
            train_metrics = utils.calculate_metrics_for_attribute(train_group1_matrix, train_group2_matrix)
            val_metrics = utils.calculate_metrics_for_attribute(val_group1_matrix, val_group2_matrix)

            print(f'\nAttribute {attr_name} Metrics:')
            print(f'  Train      - Group 1 Accuracy: {train_metrics[0]:.4f}, Group 2 Accuracy: {train_metrics[1]:.4f}, '
                  f'Total Accuracy: {train_metrics[2]:.4f}, Equalized Odds: {train_metrics[3]:.4f}')
            print(f'  Validation - Group 1 Accuracy: {val_metrics[0]:.4f}, Group 2 Accuracy: {val_metrics[1]:.4f}, '
                  f'Total Accuracy: {val_metrics[2]:.4f}, Equalized Odds: {val_metrics[3]:.4f}')

    train_epoch_conf_matrices = torch.stack(train_epoch_conf_matrices)
    val_epoch_conf_matrices = torch.stack(val_epoch_conf_matrices)

    # Save the optimized base image
    torch.save(applier.get(), save_path / "adversarial_base.pt")

    return train_epoch_conf_matrices, val_epoch_conf_matrices
    
def main(config):
    setup_start = time.perf_counter()
    save_path = Path(config['training']['save_path'])
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"PyTorch Version: {torch.__version__}")

    if config['training']['use_cuda']:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['training']['gpu_id'])
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but 'use_cuda' is set to True in the configuration.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    utils.set_seed(config['training']['random_seed'])

    # Setup model
    num_attributes = config['model']['num_attributes']
    model_path = config['model']['model_path']
    model = BinaryModel(num_attributes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Setup dataloader
    if config['dataset']['name'] == 'celeba':
        loader_function = create_celeba_data_loaders
    elif config['dataset']['name'] == 'fairface':
        loader_function = create_fairface_data_loaders
    else:
        raise ValueError("Invalid dataset name")
    train_loader, val_loader = loader_function(
        selected_attrs=config['dataset']['selected_attrs'] + [config['dataset']['protected_attr']],
        batch_size=config['training']['batch_size']
    )
    setup_end = time.perf_counter()
    print(f"Setup Time: {setup_end - setup_start:.4f} seconds")

    # Train the adversarial perturbation
    attack_params = {
        'epoch': config['training']['num_epochs'],
        'epsilon': config['attack']['epsilon'],
        'alpha': config['attack']['alpha'],
        'iters': config['attack']['iters'],
    }

    save_path = Path(config['training']['save_path'])
    base_path = config['attack']['base_path']
    attr_list = config['dataset']['selected_attrs'] # for print message to the console
    train_epoch_conf_matrices, val_epoch_conf_matrices = run_adversarial(
        model, train_loader, val_loader, attack_params, device, save_path, attr_list, base_path
    )

    # Save performance tensors
    train_perf_path = save_path / 'train_performance.pt'
    val_perf_path = save_path / 'val_performance.pt'
    torch.save(train_epoch_conf_matrices, train_perf_path)
    torch.save(val_epoch_conf_matrices, val_perf_path)

    total_time_seconds = time.perf_counter() - setup_start
    print(f"Total Time: {total_time_seconds:.2f} seconds ({total_time_seconds / 60:.2f} minutes)")

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a adversarial element')
    parser.add_argument('config_path', type=str, help='Path to the configuration YAML file')

    # Load the configuration file specified by the command-line argument
    args = parser.parse_args()
    config = utils.load_config(args.config_path)
    main(config)