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
from models.multiclass_model import MulticlassModel

def process_batch(model, images, labels, criterion, optimizer=None, device='cpu'):
    images, labels = images.to(device), labels.to(device)
    images = utils.normalize(images)
    outputs = model(images)
    if isinstance(criterion, nn.CrossEntropyLoss):
        loss = criterion(outputs, labels[:, :-1].squeeze(1))  # Exclude protected attribute
    else:
        loss = criterion(outputs, labels[:, :-1])

    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if isinstance(criterion, nn.CrossEntropyLoss):
        _, predicted = torch.max(outputs, 1)
        stats = utils.get_rights_and_wrongs_counts(predicted, labels)
    else:
        predicted = (outputs > 0.5).float()
        stats = utils.get_confusion_matrix_counts(predicted, labels)

    return loss.item(), stats

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_stats_count = None
    for i, (images, labels) in enumerate(train_loader):
        loss, stats_count = process_batch(model, images, labels, criterion, optimizer, device)

        total_loss += loss
        if all_stats_count is None:
            all_stats_count = stats_count
        else:
            all_stats_count += stats_count

    avg_loss = total_loss / (i + 1)
    return avg_loss, all_stats_count

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_stats_count = None
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            loss, stats_count = process_batch(model, images, labels, criterion, None, device)

            total_loss += loss
            if all_stats_count is None:
                all_stats_count = stats_count
            else:
                all_stats_count += stats_count

    avg_loss = total_loss / (i + 1)
    return avg_loss, all_stats_count

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

    if config['dataset']['name'] in ['celeba', 'fairface']:
        model = BinaryModel(len(config['dataset']['selected_attrs'])).to(device)
        criterion = nn.BCELoss()
    elif config['dataset']['name'] == 'ham10000':
        class_num = config['dataset']['class_number']
        model = MulticlassModel(class_num).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid dataset name: {config['dataset']['name']}")
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Setup data loader based on attack pattern
    train_loader, val_loader = utils.select_data_loader(config)
    setup_end = time.perf_counter()
    print(f"Setup Time: {setup_end - setup_start:.4f} seconds")

    num_epochs = config['training']['num_epochs']
    num_attrs = len(config['dataset'].get('selected_attrs', []))
    if isinstance(criterion, nn.CrossEntropyLoss):
        train_epoch_stats_count = torch.empty(0, 2, 2)
        val_epoch_stats_count = torch.empty(0, 2, 2)
    else:
        train_epoch_stats_count = torch.empty(0, num_attrs, 2, 4)
        val_epoch_stats_count = torch.empty(0, num_attrs, 2, 4)

    # Train and validation loop
    for epoch in range(num_epochs):
        train_loss, train_stats_count = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_stats_count = validate(model, val_loader, criterion, device)

        # Concatenate the new epoch's confusion matrices along the epoch dimension
        train_epoch_stats_count = torch.cat((train_epoch_stats_count, train_stats_count.unsqueeze(0)), dim=0)
        val_epoch_stats_count = torch.cat((val_epoch_stats_count, val_stats_count.unsqueeze(0)), dim=0)

        print(f'\nEpoch {epoch + 1} Summary:')
        print(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Print summary for each epoch
        if isinstance(criterion, nn.CrossEntropyLoss):
            utils.print_multiclass_model_summary(train_stats_count, val_stats_count, config['dataset']['selected_attr'])
        else:
            utils.print_binary_model_summary(train_stats_count, val_stats_count, config['dataset']['selected_attrs'])

        # Save model checkpoint
        checkpoint_path = save_path / f'checkpoint_epoch_{epoch+1:04d}.pth'
        torch.save(model.state_dict(), checkpoint_path)

    # Save performance tensors
    train_perf_path = save_path / 'train_performance.pt'
    val_perf_path = save_path / 'val_performance.pt'
    torch.save(train_epoch_stats_count, train_perf_path)
    torch.save(val_epoch_stats_count, val_perf_path)

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
