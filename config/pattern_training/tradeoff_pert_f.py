import os
import yaml

# Define the base configuration dictionary
base_config = {
    'training': {
        'batch_size': 128,
        'final_epoch': 30,
        'random_seed': 2665,
        'scheduler': None,
        'use_cuda': True,
        'gpu_setting': ''
    },
    'model': {
        'num_attributes': 1,
        'model_path': ''
    },
    'attack': {
        'pattern_type': 'perturbation',
        'fairness_criteria': 'equalized odds',
        'method': '',
        'alpha': 0.001,
        'iters': 1,
        'base_path': None,
        'epsilon': 0,
        'frame_thickness': 0.05,
        'gamma_adjustment': '',
        'gamma': 0,
        'gamma_adjust_factor': 0,
        'accuracy_goal': 0
    },
    'dataset': {
        'task_name': '',
        'training_schema': 'pattern',
        'name': '',
        'type': 'binary',
        'balanced': True,
        'selected_attrs': [],
        'protected_attr': '',
        'num_outputs': 1
    }
}

# Attributes info with model paths and accuracy goals for different stages
attributes_info = {
    'age': {
        'dataset_name': 'fairface',
        'protected_attr': 'race',
        'short_name': 'ff',
        'gpu_setting': '0',
        'model_path': '/tmp2/pfe/model/ff_b128/checkpoint_epoch_0006.pth',
        'initial_accuracy_goal': 0.82,
        'accuracy_goals_stage_2': [0.76, 0.74, 0.72, 0.70, 0.68],
    }
}

# Define attack method abbreviations
attack_method_abbr = {
    'fairness constraint': 'bcefc',
    'perturbed fairness constraint': 'bcepfc',
    'EquiMask fairness constraint': 'emfc',
    'EquiMask perturbed fairness constraint': 'empfc'
}

# Define variations for specific fields
epsilon_values = {
    # '4': 0.0157, '6': 0.0235, '8': 0.0314, '10': 0.0392,
    # '12': 0.0471, '14': 0.0549, '16': 0.0627, '18': 0.0706,
    '20': 0.0784, '24': 0.0941, '28': 0.1098, '32': 0.1255
}
# gamma_values_stage_1 = [10, 5, 1, 0.1, 0.01, 0.001]
gamma_values_stage_1 = [5, 1, 0.1, 0.01, 0.001]
gamma_adjust_factors_stage_1 = [0.01]
gamma_adjust_factors_stage_3 = [0.001, 0.005, 0.01, 0.05, 0.1]

# Function to save the configuration to a YAML file
def save_config_to_yaml(config, directory, file_name):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

# Function to format values
def format_value(value, total_length=7):
    # Convert the value to a string with 'd' instead of '.'
    formatted_value = f"{value:.2f}".replace('.', 'd')
    return formatted_value.rjust(total_length, '0')

# Stage 1: Find the initial gamma value
def generate_configs_stage_1():
    config_counter = 0
    for attribute, info in attributes_info.items():
        dataset_name = info['dataset_name']
        protected_attr = info['protected_attr']
        short_name = info['short_name']
        model_path = info['model_path']
        gpu_setting = info['gpu_setting']
        initial_accuracy_goal = info['initial_accuracy_goal']
        max_length = max(len(f"{value:.2f}".replace('.', 'd')) for value in gamma_values_stage_1)

        for attack_method, method_abbr in attack_method_abbr.items():
            for pixel_value, epsilon in epsilon_values.items():
                for gamma in gamma_values_stage_1:
                    formatted_gamma = format_value(gamma, max_length)
                    config = base_config.copy()
                    config['training']['save_path'] = f"/tmp2/pfe/pattern/pert/tradeoff/{pixel_value}/{method_abbr}/{short_name}_g_{formatted_gamma}"
                    config['training']['gpu_setting'] = gpu_setting
                    config['model']['model_path'] = model_path
                    config['attack']['method'] = attack_method
                    config['attack']['epsilon'] = epsilon
                    config['attack']['gamma_adjustment'] = 'constant'
                    config['attack']['gamma'] = gamma
                    config['attack']['gamma_adjust_factor'] = 0.01
                    config['attack']['accuracy_goal'] = initial_accuracy_goal
                    config['dataset']['task_name'] = attribute
                    config['dataset']['name'] = dataset_name
                    config['dataset']['selected_attrs'] = [attribute]
                    config['dataset']['protected_attr'] = protected_attr

                    directory = f"perturbation/tradeoff/{pixel_value}/{method_abbr}"
                    file_name = f"{short_name}_g_{formatted_gamma}.yml"
                    save_config_to_yaml(config, directory, file_name)
                    config_counter += 1

    print(f"Generated {config_counter} configuration files for Stage 1.")

generate_configs_stage_1()