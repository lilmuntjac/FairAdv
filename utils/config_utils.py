import os
import yaml
import numpy as np

import torch

def load_config(config_path):
    """Load a YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def config_env(config, title='training'):
    """Configures the device for PyTorch and initializes random seeds for reproducibility."""
    print(f"PyTorch Version: {torch.__version__}")

    # Set the CUDA device environment variable based on the configuration.
    use_cuda = config[title].get('use_cuda', False)
    gpu_setting = str(config[title].get('gpu_setting', '0'))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_setting

    # Validate CUDA availability if requested.
    if use_cuda:
        print(f"GPU setting: {gpu_setting}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is requested but not available; "
                   "check CUDA_VISIBLE_DEVICES or GPU availability.")
        device = 'cuda'
    else:
        device = 'cpu'
        print("Using CPU for computations.")

    # Set random seeds to ensure experiments can be reproducible.
    seed = config[title].get('random_seed', 2665)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    return device
