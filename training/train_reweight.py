import time
from pathlib import Path

import torch

from data.loaders.samplers import SeededSampler
from utils.metrics_utils import (
    normalize, get_confusion_matrix_counts, get_rights_and_wrongs_counts, 
    print_binary_model_summary, print_multiclass_model_summary
)

class ReWeightTrainer:
    def __init__(self, config, train_loader, val_loader,
                 model, criterion, optimizer, scheduler, save_path, device='cpu'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.model_type = config['dataset']['type']
        self.task_name = config['dataset'].get('task_name', 'unspecified')
        self.criterion = criterion 
        self.optimizer = optimizer 
        self.scheduler = scheduler
        self.save_path = Path(save_path) # Ensure it's a Path object
        self.device = device
        # reweight (Label Bias Correcting)
        self.iteration = config['reweight'].get('iteration', 10)
        self.eta = config['reweight'].get('eta', 0.001)
        protected_attr = config['dataset']['protected_attr']
        if isinstance(protected_attr, str):
            num_protected_attributes = 1
        elif isinstance(protected_attr, list):
            num_protected_attributes = len(protected_attr)
        else:
            raise ValueError("The 'protected_attr' should be either a string or a list.")
        self.multipliers = torch.zeros(2 * num_protected_attributes, device=self.device)
    

    def run(self, _, _, total_train_stats, total_val_stats):
        # This trainer has iterations outside the epoch and only saves per iteration. 
        # Be careful not to confuse it with other trainers.
        # The start_epoch and final_epoch argument would not work
        total_start_time = time.perf_counter()

        for iteration in range(self.iteration):
            print(f"Iteration {iteration + 1}/{self.iteration}")
            start_time = time.perf_counter()

            



        