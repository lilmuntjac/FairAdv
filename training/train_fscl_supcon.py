import time
from pathlib import Path

import torch
import torch.nn as nn

import utils.utils as utils

class FSCLSupConTrainer:
    def __init__(self, config, train_loader, val_loader,
                 model, criterion, optimier, scheduler, save_path, device='cpu'):
        self.train_loader = train_loader