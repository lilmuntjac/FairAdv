import time
import argparse
from pathlib import Path

from utils.config_utils import load_config, config_env
from utils.training_utils import (
    setup_training_components, load_training_components, setup_stats_tensors, 
    select_data_loader, select_trainer
)

def main(config):
    setup_start = time.perf_counter()
    save_path = Path(config['training']['save_path'])
    save_path.mkdir(parents=True, exist_ok=True)

    device = config_env(config)

    # Set up the classes for the model and other components.
    # Load checkpoints and statistics if resuming model training; otherwise, initialize the statistics.
    # If this setup is not for training the model, please load the pre-trained model elsewhere.
    model, optimizer, criterion, scheduler = setup_training_components(config, device)
    passed_epoch = load_training_components(config, model, optimizer, scheduler, device)
    total_train_stats, total_val_stats = setup_stats_tensors(config, passed_epoch)

    # Set up the data loader based on the different training types.
    train_loader, val_loader = select_data_loader(config)
    setup_end = time.perf_counter()
    print(f"Setup Time: {setup_end - setup_start:.4f} seconds")

    # Train and validation loop
    start_epoch = passed_epoch + 1
    final_epoch = config['training']['final_epoch']

    TrainerClass = select_trainer(config)
    trainer = TrainerClass(config, train_loader, val_loader,
                           model, criterion, optimizer, scheduler, save_path, device)
    trainer.run(start_epoch, final_epoch, total_train_stats, total_val_stats)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config_path', type=str, help='Path to the configuration YAML file')

    # Load the configuration file specified by the command-line argument
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config)
