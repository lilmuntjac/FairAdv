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

def print_epoch_summary(epoch, train_conf, val_conf, attr_list):
    # Compute and print metrics for each attribute
    print(f'\nEpoch {epoch + 1} Summary:')
    for attr_index, attr_name in enumerate(attr_list):
        # Extract confusion matrices for the current attribute and epoch
        train_group1_matrix = train_conf[attr_index, 0]
        train_group2_matrix = train_conf[attr_index, 1]
        val_group1_matrix = val_conf[attr_index, 0]
        val_group2_matrix = val_conf[attr_index, 1]

        # Calculate metrics for the current attribute
        train_metrics = utils.calculate_metrics_for_attribute(train_group1_matrix, train_group2_matrix)
        val_metrics = utils.calculate_metrics_for_attribute(val_group1_matrix, val_group2_matrix)
        print(f'\nAttribute {attr_name} Metrics:')
        print(f'  Train      - Group 1 Accuracy: {train_metrics[0]:.4f}, Group 2 Accuracy: {train_metrics[1]:.4f}, '
              f'Total Accuracy: {train_metrics[2]:.4f}, Equalized Odds: {train_metrics[3]:.4f}')
        print(f'  Validation - Group 1 Accuracy: {val_metrics[0]:.4f}, Group 2 Accuracy: {val_metrics[1]:.4f}, '
              f'Total Accuracy: {val_metrics[2]:.4f}, Equalized Odds: {val_metrics[3]:.4f}')

def run_adversarial(model, train_loader, val_loader, applier, attack_params, device, save_path, attr_list):
    """Run adversarial attack and evaluate on train and validation datasets."""
    train_epoch_conf_matrices = []
    val_epoch_conf_matrices = []
    model.eval()

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

        print_epoch_summary(epoch, train_conf_matrices, val_conf_matrices, attr_list)
        # Save the optimized base image
        torch.save(applier.get(), save_path / f"base_{epoch + 1:04d}.pt")

    train_epoch_conf_matrices = torch.stack(train_epoch_conf_matrices)
    val_epoch_conf_matrices = torch.stack(val_epoch_conf_matrices)

    return train_epoch_conf_matrices, val_epoch_conf_matrices
    
def main(config):
    setup_start = time.perf_counter()
    save_path = Path(config['training']['save_path'])
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"PyTorch Version: {torch.__version__}")

    device = utils.config_env(config)

    # Setup model
    num_attributes = config['model']['num_attributes']
    model_path = config['model']['model_path']
    model = utils.select_model(config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Setup data loader based on attack pattern
    train_loader, val_loader = utils.select_data_loader(config)
    setup_end = time.perf_counter()
    print(f"Setup Time: {setup_end - setup_start:.4f} seconds")

    # Train the adversarial perturbation
    save_path = Path(config['training']['save_path'])
    applier = utils.select_applier(config, device=device)

    attack_params = {
        'epoch': config['training']['num_epochs'],
        'epsilon': config['attack']['epsilon'],
        'alpha': config['attack']['alpha'],
        'iters': config['attack']['iters'],
    }

    attr_list = config['dataset']['selected_attrs'] # for print message to the console
    train_epoch_conf_matrices, val_epoch_conf_matrices = run_adversarial(
        model, train_loader, val_loader, applier, attack_params, device, save_path, attr_list,
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