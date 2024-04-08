import os
import time
import argparse
import numpy as np
from pathlib import Path

import utils.utils as utils
from utils.model_utils import setup_model_and_optimizer, setup_training_environment

def main(config):
    setup_start = time.perf_counter()
    save_path = Path(config['training']['save_path'])
    save_path.mkdir(parents=True, exist_ok=True)

    device = utils.config_env(config)

    # Set up the class of the model and other components.
    # Load checkpoints and stats if resuming the model training; otherwise, initialize the stats.
    # If it's not for training the model, please load the pre-trained model elsewhere.
    model, criterion, optimizer, scheduler = setup_model_and_optimizer(config, device)
    (passed_epoch, total_train_stats, 
     total_val_stats) = setup_training_environment(config, model, optimizer, scheduler, device)

    # Set up the data loader based on the different training types.
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
