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

from utils.config_utils import load_config

def compute_epoch_stats(stats, stats_type, epoch_index):
    if stats_type == 'binary':
        epoch_stats = stats[epoch_index, :, :]
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

        return {
            'group1_accuracy': group1_accuracy,
            'group2_accuracy': group2_accuracy,
            'total_accuracy': total_accuracy,
            'equalized_odds': equalized_odds
        }
    elif stats_type == 'mult-class':
        epoch_stats = stats[epoch_index, :, :]
        R1, W1 = epoch_stats[0, 0], epoch_stats[0, 1]
        R2, W2 = epoch_stats[1, 0], epoch_stats[1, 1]

        # Accuracies
        group1_accuracy = R1 / (R1 + W1) if (R1 + W1) > 0 else 0
        group2_accuracy = R2 / (R2 + W2) if (R2 + W2) > 0 else 0

        total_instances = R1 + R2 + W1 + W2
        total_accuracy = (R1 + R2) / total_instances if total_instances > 0 else 0
        # Difference in accuracy
        differences_acc = abs(group1_accuracy-group2_accuracy)
        return {
            'group1_accuracy': group1_accuracy,
            'group2_accuracy': group2_accuracy,
            'total_accuracy': total_accuracy,
            'differences_acc': differences_acc
        }
    else:
        raise ValueError(f"Invalid stats type: {stats_type}")

def main(config):
    time_start = time.perf_counter()

    # Load the statistics
    stats_file_path = config['analysis']['stats_file_path']
    stats_type = config['analysis']['type']
    task_name = config['analysis'].get('task_name', 'unspecified')
    stats =  torch.load(stats_file_path)
    final_epoch, _, val_stats ,val_robust = stats['epoch'], stats['train'], stats['val'], stats['val_robust']

    print(f"\nAnalyzing: {task_name}")
    best_epoch, best_score, scores = None, -2, []
    initial_status = config['analysis'].get('initial_status', {})
    if not initial_status:
        raise ValueError("The config must contain the 'initial_status' ")
    
    for epoch_index in range(val_stats.shape[0]):
        current_status = compute_epoch_stats(val_stats, stats_type, epoch_index)
        current_robust_status = compute_epoch_stats(val_robust, stats_type, epoch_index)
        # Calculate score
        accuracy_loss = initial_status['total_accuracy'] - current_status['total_accuracy']
        robust_accuracy_loss = initial_status['total_accuracy'] - current_robust_status['total_accuracy']
        
        combine_loss = accuracy_loss + robust_accuracy_loss
        print(f"Epoch {epoch_index}: Clean Accuracy: {current_status['total_accuracy']:.4f}, Robust Accuracy: {current_robust_status['total_accuracy']:.4f}, Combined loss: {combine_loss:.4f}")
        scores.append(-combine_loss)

     # Identify best epoch
    best_epoch_index = np.argmax(scores)
    best_epoch = best_epoch_index + 1
    best_score = scores[best_epoch_index]
    best_epoch_stats_clean = compute_epoch_stats(val_stats, stats_type, best_epoch_index)
    best_epoch_stats_robust = compute_epoch_stats(val_robust, stats_type, best_epoch_index)
    print(f"\nBest Epoch: {best_epoch} with Score: {best_score:.4f}")
    print(f"Clean validation accuracy:")
    for key in best_epoch_stats_clean:
        print(f'{key}: {best_epoch_stats_clean[key].item():.4f}')
    print(f"\nRobust validation accuracy:")
    for key in best_epoch_stats_robust:
        print(f'{key}: {best_epoch_stats_robust[key].item():.4f}')

    print(f"Total Time: {time.perf_counter() - time_start:.4f} seconds")

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config_path', type=str, help='Path to the configuration YAML file')

    # Load the configuration file specified by the command-line argument
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config)