import os
import time
import argparse
import numpy as np
from pathlib import Path

import utils.utils as utils
from utils.model_utils import setup_model_and_optimizer, load_checkpoint_and_stats

def main(config):
    setup_start = time.perf_counter()
    save_path = Path(config['training']['save_path'])
    save_path.mkdir(parents=True, exist_ok=True)

    device = utils.config_env(config)

    model, criterion, optimizer, scheduler = setup_model_and_optimizer(config, device)
    (passed_epoch, total_train_stats, 
     total_val_stats) = load_checkpoint_and_stats(config, model, optimizer, scheduler, device)

    # Setup data loader based on attack pattern
    train_loader, val_loader = utils.select_data_loader(config)
    setup_end = time.perf_counter()
    print(f"Setup Time: {setup_end - setup_start:.4f} seconds")

    # Train and validation loop
    start_epoch = passed_epoch + 1
    final_epoch = config['training']['final_epoch']

    TrainerClass = utils.select_trainer(config)
    trainer = TrainerClass(config, train_loader, val_loader,
                           model, criterion, optimizer, scheduler, save_path, device)
    trainer.run(start_epoch, final_epoch, total_train_stats, total_val_stats)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a CelebA model')
    parser.add_argument('config_path', type=str, help='Path to the configuration YAML file')

    # Load the configuration file specified by the command-line argument
    args = parser.parse_args()
    config = utils.load_config(args.config_path)
    main(config)
