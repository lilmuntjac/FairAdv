import time
from pathlib import Path

import torch

from utils.metrics_utils import (
    normalize, get_confusion_matrix_counts, get_rights_and_wrongs_counts, 
    print_binary_model_summary, print_multiclass_model_summary
)

class FHSICTrainer:
    def __init__(self, config, train_loader, val_loader,
                 model, criterion, optimizer, scheduler, save_path, device='cpu'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.model_type = config['dataset']['type']
        self.task_name = config['dataset'].get('task_name', 'unspecified')
        self.criterion = criterion # unused
        self.optimizer = optimizer # unused
        self.scheduler = scheduler
        self.save_path = Path(save_path) # Ensure it's a Path object
        self.device = device

    def run(self, start_epoch, final_epoch, total_train_stats, total_val_stats):
        if start_epoch > final_epoch:
            print("Start epoch must be less than final epoch.")
            return
        total_start_time = time.perf_counter()