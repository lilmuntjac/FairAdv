import os
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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import utils.utils as utils
from models.binary_model import BinaryModel

cm_cell_def = ['TP', 'FN', 'FP', 'TN']

def plot_confusion_matrices(cm_per_group, cm_for_all, accuracy, title, ax):
    im = ax.imshow(cm_per_group, cmap=plt.cm.cool, vmin=0, vmax=1)
    ax.set_title(f'{title} Accuracy: {accuracy:.2%}')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel('True Class')
    ax.set_ylabel('Predicted Class')
    ax.set_xticklabels(['Positive', 'Negative'])
    ax.set_yticklabels(['Positive', 'Negative'], rotation=90, va='center')
    for i, cell_type in enumerate(cm_cell_def):
        dist_per_group = cm_per_group.flatten()[i].item()
        dist_for_all = cm_for_all.flatten()[i].item()
        info = f'{cell_type}\n{dist_per_group:.2%}({title})\n{dist_for_all:.2%}(All)'
        ax.text(i % 2, i // 2, s=info, va='center', ha='center')
    return im

def get_cm_dist(model, loader, device, attr_list, save_path):
    # get confusion matrices from a pre-trained model
    model.eval()
    conf_matrices = None
    with torch.no_grad():
        for i, (images, labels) in enumerate (loader):
            images, labels = images.to(device), labels.to(device)
            images = utils.normalize(images)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            conf_matrix = utils.get_confusion_matrix_counts(predicted, labels)
            if conf_matrices is None:
                conf_matrices = conf_matrix
            else:
                conf_matrices += conf_matrix
    # Create one image per attributes
    for attr_index, attr_name in enumerate(attr_list):
        # Get the basic information about this attribute
        group1_matrix = conf_matrices[attr_index, 0]
        group2_matrix = conf_matrices[attr_index, 1]
        metrics = utils.calculate_metrics_for_attribute(group1_matrix, group2_matrix)

        fig, axs  = plt.subplots(1,2)
        fig.suptitle(
            f'{attr_name} Metrics - Accuracy: {metrics[2]:.4f}, Equalized Odds: {metrics[3]:.4f}',
            y=0.85
        )
        # Draw a picture for each of the different protective attributes
        ims = list()
        is_suitable = True
        for group_idx, group in enumerate(['F', 'M']):
            normalize_cm_per_group = conf_matrices[attr_index, group_idx, :4].reshape(2, 2)/\
            torch.sum(conf_matrices[attr_index, group_idx, :4])
            normalize_cm_for_all = conf_matrices[attr_index, group_idx, :4].reshape(2, 2)/\
            torch.sum(conf_matrices[attr_index, :2, :4])
            # Draw the individual subplot
            ims.append(plot_confusion_matrices(normalize_cm_per_group, normalize_cm_for_all,
                                               metrics[group_idx], group, axs[group_idx]))
            # Check the attribute quaility
            threshold = 0.05
            if (normalize_cm_per_group[0,0] < threshold or normalize_cm_per_group[1,1] < threshold or
                    normalize_cm_for_all[0,0] < threshold or normalize_cm_for_all[0,0] < threshold):
                is_suitable = False
        if is_suitable:
            print(f'Attribute suitable for revise: {attr_name}')

        for i, im in enumerate(ims):
            fig.colorbar(im, ax=axs[i], fraction=0.045)

        fig_path = save_path / (attr_name + '.png')
        fig.tight_layout()
        fig.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

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

    # Setup model (binary)
    num_attributes = config['model']['num_attributes']
    model_path = config['model']['model_path']
    model = BinaryModel(num_attributes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Setup data loader based on attack pattern
    train_loader, val_loader = utils.select_data_loader(config)
    setup_end = time.perf_counter()
    print(f"Setup Time: {setup_end - setup_start:.4f} seconds")

    # Directory to save transformed images
    save_path = Path(config['training']['save_path'])
    save_path.mkdir(parents=True, exist_ok=True)

    attr_list = config['dataset']['selected_attrs'] # for print message to the console
    get_cm_dist(model, train_loader, device, attr_list, save_path)
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