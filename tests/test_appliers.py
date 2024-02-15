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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image

import utils.utils as utils

def apply_pattern_and_save(loader, applier, save_path, device):
    eyeglasses_applier_class = 'EyeglassesApplier'  # Class name as a string

    for batch_idx, batch in enumerate(loader):
        if len(batch) == 3: # the loader include theta
            images, theta, _ = batch
            images, theta = images.to(device), theta.to(device)
            if type(applier).__name__ == eyeglasses_applier_class:
                forged_images = applier.apply(images, theta)
            else:
                forged_images = applier.apply(images)
        elif len(batch) == 2: # normal loader
            images, _ = batch
            images = images.to(device)
            forged_images = applier.apply(images)
        else:
            raise ValueError("Unexpected data format from the loader")
        
        # Save the transformed images
        for image_idx, img in enumerate(forged_images):
            save_image(img, save_path / f'{batch_idx}_{image_idx:04d}.png')

        if batch_idx >= 0:  # Process only the first batch
            break

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

    # Setup data loader based on attack pattern
    loader_function = utils.select_data_loader(config)
    train_loader, val_loader = utils.select_data_loader(config)
    setup_end = time.perf_counter()
    print(f"Setup Time: {setup_end - setup_start:.4f} seconds")

    # Directory to save transformed images
    save_path = Path(config['training']['save_path']) / "forged_images"
    save_path.mkdir(parents=True, exist_ok=True)

    applier = utils.select_applier(config, device=device)
    apply_pattern_and_save(val_loader, applier, save_path, device)

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