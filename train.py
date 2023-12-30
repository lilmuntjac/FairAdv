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
from models.binary_model import BinaryModel
from utils.data_loader import create_celeba_data_loaders

def process_batch(model, images, labels, criterion, optimizer=None, device='cpu'):
    images, labels = images.to(device), labels.to(device)
    images = utils.normalize(images)
    outputs = model(images)
    loss = criterion(outputs, labels[:, :-1])  # Exclude protected attribute

    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predicted = (outputs > 0.5).float()
    conf_matrix = utils.get_confusion_matrix_counts(predicted, labels)

    return loss.item(), conf_matrix

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_conf_matrices = None
    for i, (images, labels) in enumerate(train_loader):
        loss, conf_matrix = process_batch(model, images, labels, criterion, optimizer, device)

        total_loss += loss
        if all_conf_matrices is None:
            all_conf_matrices = conf_matrix
        else:
            all_conf_matrices += conf_matrix

    avg_loss = total_loss / (i + 1)
    return avg_loss, all_conf_matrices

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_conf_matrices = None
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            loss, conf_matrix = process_batch(model, images, labels, criterion, None, device)

            total_loss += loss
            if all_conf_matrices is None:
                all_conf_matrices = conf_matrix
            else:
                all_conf_matrices += conf_matrix

    avg_loss = total_loss / (i + 1)
    return avg_loss, all_conf_matrices

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

    model = BinaryModel(len(config['dataset']['selected_attrs'])).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    train_loader, val_loader = create_celeba_data_loaders(
        selected_attrs=config['dataset']['selected_attrs'] + [config['dataset']['protected_attr']],
        batch_size=config['training']['batch_size']
    )
    setup_end = time.perf_counter()
    print(f"Setup Time: {setup_end - setup_start:.4f} seconds")

    num_epochs = config['training']['num_epochs']
    num_attrs = len(config['dataset']['selected_attrs'])
    train_epoch_conf_matrices = torch.empty(0, num_attrs, 2, 4)
    val_epoch_conf_matrices = torch.empty(0, num_attrs, 2, 4)

    # Train and validation loop
    for epoch in range(num_epochs):
        train_loss, train_conf_matrices = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_conf_matrices = validate(model, val_loader, criterion, device)

        # Concatenate the new epoch's confusion matrices along the epoch dimension
        train_epoch_conf_matrices = torch.cat((train_epoch_conf_matrices, train_conf_matrices.unsqueeze(0)), dim=0)
        val_epoch_conf_matrices = torch.cat((val_epoch_conf_matrices, val_conf_matrices.unsqueeze(0)), dim=0)

        print(f'\nEpoch {epoch + 1} Summary:')
        print(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Compute and print metrics for each attribute
        for attr_index, attr_name in enumerate(config['dataset']['selected_attrs']):
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
            
        # Save model checkpoint
        checkpoint_path = save_path / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)

    # Save performance tensors
    train_perf_path = save_path / 'train_performance.pt'
    val_perf_path = save_path / 'val_performance.pt'
    torch.save(train_epoch_conf_matrices, train_perf_path)
    torch.save(val_epoch_conf_matrices, val_perf_path)

    total_time_seconds = time.perf_counter() - setup_start
    print(f"Total Time: {total_time_seconds:.2f} seconds ({total_time_seconds / 60:.2f} minutes)")


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a CelebA model')
    parser.add_argument('config_path', type=str, help='Path to the configuration YAML file')

    # Load the configuration file specified by the command-line argument
    args = parser.parse_args()
    config = utils.load_config(args.config_path)
    main(config)
