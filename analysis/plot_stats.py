import sys
import time
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add the project root to the Python path
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent  # Adjust this path to point to the project root
sys.path.append(str(project_root))

import torch

import utils.utils as utils

def calculate_binary_model_metrics(stats):
    metrics = {
        'group1_accuracies': [], 'group2_accuracies': [],
        'total_accuracies': [], 'equalized_odds_list': []
    }

    # Iterate through each epoch
    for epoch_stats in stats:
        TP1, FP1, FN1, TN1 = epoch_stats[0, :]
        TP2, FP2, FN2, TN2 = epoch_stats[1, :]

        # Accuracies
        group1_denominator = TP1 + FP1 + FN1 + TN1
        group1_accuracy = (TP1 + TN1) / group1_denominator if group1_denominator > 0 else 0
        group2_denominator = TP2 + FP2 + FN2 + TN2
        group2_accuracy = (TP2 + TN2) / group2_denominator if group2_denominator > 0 else 0
        
        total_denominator = group1_denominator + group2_denominator
        total_accuracy = (TP1 + TN1 + TP2 + TN2) / total_denominator if total_denominator > 0 else 0
        # Equalized odds
        TPR1 = TP1 / (TP1 + FN1) if TP1 + FN1 > 0 else 0
        TPR2 = TP2 / (TP2 + FN2) if TP2 + FN2 > 0 else 0
        FPR1 = FP1 / (FP1 + TN1) if FP1 + TN1 > 0 else 0
        FPR2 = FP2 / (FP2 + TN2) if FP2 + TN2 > 0 else 0
        equalized_odds = abs(TPR1 - TPR2) + abs(FPR1 - FPR2)

        metrics['group1_accuracies'].append(group1_accuracy)
        metrics['group2_accuracies'].append(group2_accuracy)
        metrics['total_accuracies'].append(total_accuracy)
        metrics['equalized_odds_list'].append(equalized_odds)

    return metrics

def calculate_multiclass_model_metrics(stats):
    metrics = {
        'group1_accuracies': [], 'group2_accuracies': [],
        'total_accuracies': [], 'differences_acc_list': []
    }

    # Iterate through each epoch
    for epoch_stats in stats:
        R1, W1 = epoch_stats[0, 0], epoch_stats[0, 1]
        R2, W2 = epoch_stats[1, 0], epoch_stats[1, 1]

        # Accuracies
        group1_accuracy = R1 / (R1 + W1) if (R1 + W1) > 0 else 0
        group2_accuracy = R2 / (R2 + W2) if (R2 + W2) > 0 else 0

        total_instances = R1 + R2 + W1 + W2
        total_accuracy = (R1 + R2) / total_instances if total_instances > 0 else 0
        # Difference in accuracy
        differences_acc = abs(group1_accuracy-group2_accuracy)

        metrics['group1_accuracies'].append(group1_accuracy)
        metrics['group2_accuracies'].append(group2_accuracy)
        metrics['total_accuracies'].append(total_accuracy)
        metrics['differences_acc_list'].append(differences_acc)

    return metrics

def plot_data(train_metrics, val_metrics, epoch_range, metric_key):
    marker_style_train = 'o'
    marker_style_val = 's'
    # Check the metric key and plot accordingly
    # Adjust or extend this logic based on the metrics you need to plot
    if metric_key == 'accuracy':
        plt.plot(epoch_range, train_metrics['total_accuracies'], 
                 label=f'Train Total Accuracy', marker=marker_style_train)
        plt.plot(epoch_range, val_metrics['total_accuracies'], '--', 
                 label=f'Validation Total Accuracy', marker=marker_style_val)
    elif metric_key == 'group_accuracy':
        plt.plot(epoch_range, train_metrics['group1_accuracies'], 
                 label=f'Train Group 1 Accuracy', marker=marker_style_train)
        plt.plot(epoch_range, train_metrics['group2_accuracies'], 
                 label=f'Train Group 2 Accuracy', marker=marker_style_train)
        plt.plot(epoch_range, val_metrics['group1_accuracies'], '--', 
                 label=f'Validation Group 1 Accuracy', marker=marker_style_val)
        plt.plot(epoch_range, val_metrics['group2_accuracies'], '--', 
                 label=f'Validation Group 2 Accuracy', marker=marker_style_val)
    elif metric_key == 'equalized_odds':
        plt.plot(epoch_range, train_metrics['equalized_odds_list'], 
                 label=f'Train Equalized Odds', marker=marker_style_train)
        plt.plot(epoch_range, val_metrics['equalized_odds_list'], 
                 label=f'Validation Equalized Odds', marker=marker_style_val)
    elif metric_key == 'equalized_accuracy':
        plt.plot(epoch_range, train_metrics['differences_acc_list'], 
                 label=f'Train Equalized Accuracy', marker=marker_style_train)
        plt.plot(epoch_range, val_metrics['differences_acc_list'], 
                 label=f'Validation Equalized Accuracy', marker=marker_style_val)
    else:
        raise NotImplementedError(f"Plotting for metric_key '{metric_key}' is not implemented.")
            
def plot_attribute_stats(train_stats, val_stats, stats_type, epoch_range, attr_names, metric_keys, save_path):
    for attr_index, attr_name in enumerate(attr_names):
        if stats_type == 'binary':
            train_metrics = calculate_binary_model_metrics(train_stats[:, attr_index, :, :])
            val_metrics = calculate_binary_model_metrics(val_stats[:, attr_index, :, :])
        elif stats_type == 'mult-class':
            train_metrics = calculate_multiclass_model_metrics(train_stats)
            val_metrics = calculate_multiclass_model_metrics(val_stats)
        else:
            raise ValueError(f"Invalid stats type: {stats_type}")
        
        for metric_key in metric_keys:
            plt.figure(figsize=(20, 6))

            formatted_metric_name = ' '.join(
                word.capitalize() for word in metric_key.replace('-', ' ').replace('_', ' ').split()
            )
            # First subplot with Matplotlib determined y-axis limits
            plt.subplot(1, 2, 1)
            plot_data(train_metrics, val_metrics, epoch_range, metric_key)
            plt.xlabel('Epoch')
            plt.ylabel(formatted_metric_name)
            plt.legend()
            plt.title(f'{formatted_metric_name} over Epochs for {attr_name} (Auto y-lim)', fontsize=14)

            # Second subplot with y-axis limits set to [0, 1]
            plt.subplot(1, 2, 2)
            plot_data(train_metrics, val_metrics, epoch_range, metric_key)
            plt.ylim([0, 1])
            plt.xlabel('Epoch')
            plt.ylabel(formatted_metric_name)
            plt.legend()
            plt.title(f'{formatted_metric_name} over Epochs for {attr_name} (y-lim: 0 to 1)', fontsize=14)
        
            # Save the plot for the attribute with all specified metrics included
            filename = f"{attr_name.replace(' ', '_')}_{metric_key.replace(' ', '_')}.png"
            plt.savefig(save_path / filename, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            print(f"Plot saved as: {save_path / filename}")

def main(config):
    time_start = time.perf_counter()

    # Directory for saving plots
    save_path = Path(config['plotting']['save_path'])
    save_path.mkdir(parents=True, exist_ok=True)

    # Load the statistics
    stats_file_path = config['plotting']['stats_file_path']
    stats_type = config['plotting']['type']
    if stats_type not in ['binary', 'mult-class']:
        raise ValueError(f"Invalid stats type: {stats_type}")
    metrics = config['plotting'].get('metrics', [])
    attr_names = config['plotting'].get('attributes', 'unspecified')
    if not Path(stats_file_path).is_file():
        raise FileNotFoundError(f"Stats file not found: {stats_file_path}")
    stat =  torch.load(stats_file_path)
    final_epoch, total_train_stats, total_val_stats = stat['epoch'], stat['train'], stat['val']

    # Retrieve the stats tensor based on epoch settings
    first_epoch_num = config['plotting'].get('first_epoch_num', 1)
    if first_epoch_num == 0: # only for method that record initial model status
        epoch_start = config['plotting']['epoch'].get('start', 0)
        epoch_end = config['plotting']['epoch'].get('end', total_train_stats.shape[0])
        epoch_range = range(epoch_start, epoch_end+1)
        train_stats, val_stats = total_train_stats[epoch_start:epoch_end+1], total_val_stats[epoch_start:epoch_end+1]
    elif first_epoch_num == 1:
        epoch_start = config['plotting']['epoch'].get('start', 1)
        if epoch_start < 1:
            raise ValueError(f"Epoch starts from 1 but {epoch_start} is smaller")
        epoch_end = config['plotting']['epoch'].get('end', total_train_stats.shape[0])
        epoch_range = range(epoch_start, epoch_end+1)
        train_stats, val_stats = total_train_stats[epoch_start-1:epoch_end], total_val_stats[epoch_start-1:epoch_end]
    else:
        raise ValueError(f"Epoch can only starts from 0 or 1")
    
    # plot the diagram per attribute
    plot_attribute_stats(train_stats, val_stats, stats_type, epoch_range, attr_names, metrics, save_path)

    print(f"Total Time: {time.perf_counter() - time_start:.4f} seconds")


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config_path', type=str, help='Path to the configuration YAML file')

    # Load the configuration file specified by the command-line argument
    args = parser.parse_args()
    config = utils.load_config(args.config_path)
    main(config)