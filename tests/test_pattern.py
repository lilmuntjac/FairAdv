import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Add the project root to the Python path
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent  # Adjust this path to point to the project root
sys.path.append(str(project_root))

import torch

import utils.utils as utils
from models.binary_model import BinaryModel

def validation(model, images, labels, applier):
    perturbed_images = applier.apply(images)
    perturbed_images = utils.normalize(perturbed_images)
    outputs = model(perturbed_images)
    predicted = (outputs > 0.5).float()
    conf_matrix = utils.get_confusion_matrix_counts(predicted, labels)

    return conf_matrix

def print_summary(val_conf, attr_list):
    for attr_index, attr_name in enumerate(attr_list):
        val_group1_matrix = val_conf[attr_index, 0]
        val_group2_matrix = val_conf[attr_index, 1]

        # Calculate metrics for the current attribute
        val_metrics = utils.calculate_metrics_for_attribute(val_group1_matrix, val_group2_matrix)
        print(f'\nAttribute {attr_name} Metrics:')
        print(f'  Validation - Group 1 Accuracy: {val_metrics[0]:.4f}, Group 2 Accuracy: {val_metrics[1]:.4f}, '
              f'Total Accuracy: {val_metrics[2]:.4f}, Equalized Odds: {val_metrics[3]:.4f}')

def validate_adversarial(model, val_loader, applier, device, attr_list):
    model.eval()

    val_conf_matrices = None
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            conf_matrix = validation(model, images, labels, applier)
            if val_conf_matrices is None:
                val_conf_matrices = conf_matrix
            else:
                val_conf_matrices += conf_matrix

    print_summary(val_conf_matrices, attr_list)

def main(config):
    setup_start = time.perf_counter()
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

    # Setup data loader based on attack pattern
    train_loader, val_loader = utils.select_data_loader(config)
    setup_end = time.perf_counter()
    print(f"Setup Time: {setup_end - setup_start:.4f} seconds")

    applier = utils.select_applier(config, device=device)
    attr_list = config['dataset']['selected_attrs'] # for print message to the console
    validate_adversarial(model, val_loader, applier, device, attr_list)

    total_time_seconds = time.perf_counter() - setup_start
    print(f"Total Time: {total_time_seconds:.2f} seconds ({total_time_seconds / 60:.2f} minutes)")

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('config_path', type=str, help='Path to the configuration YAML file')

    # Load the configuration file specified by the command-line argument
    args = parser.parse_args()
    config = utils.load_config(args.config_path)
    main(config)