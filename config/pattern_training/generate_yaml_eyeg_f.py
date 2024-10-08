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
        'pattern_type': 'eyeglasses',
        'fairness_criteria': 'equalized odds',
        'method': '',
        'alpha': 0.001,
        'iters': 1,
        'base_path': None,
        'epsilon': 0.063,
        'frame_thickness': 0.05,
        'gamma_adjustment': '',
        'gamma': 0,
        'gamma_adjust_factor': 0,
        'accuracy_goal': 0
    },
    'dataset': {
        'task_name': 'Age',
        'training_schema': 'pattern',
        'name': 'fairface',
        'type': 'binary',
        'balanced': True,
        'selected_attrs': ['age'],
        'protected_attr': 'race',
        'num_outputs': 1
    }
}

# Attributes info with model paths and accuracy goals for different stages
attributes_info = {
    'age': {
        'short_name': 'ff',
        'model_path': '/tmp2/pfe/model/ff_b128/checkpoint_epoch_0006.pth',
        'initial_accuracy_goal': 0.82,
        # 'accuracy_goals_stage_2': [0.86, 0.84, 0.82, 0.80, 0.78],
        # 'accuracy_goals_stage_2': [0.76, 0.74, 0.72, 0.70, 0.68],
        'accuracy_goals_stage_2': [0.80, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68],
        'gpu_setting': '1'
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
gamma_values_stage_1 = [10, 5, 1, 0.1, 0.01]
gamma_adjust_factors_stage_1 = [0.01]
gamma_adjust_factors_stage_3 = [0.001, 0.005, 0.01, 0.05, 0.1]

# Function to save the configuration to a YAML file
def save_config_to_yaml(config, directory, file_name):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

# Function to format values
def format_value(value):
    if value < 1:
        return str(value).replace('.', 'd').ljust(6, '0')
    return str(value).replace('.', 'd')

# Stage 1: Find the initial gamma value
def generate_configs_stage_1():
    config_counter = 0
    gamma_results = {}
    for attribute, info in attributes_info.items():
        short_name = info['short_name']
        model_path = info['model_path']
        initial_accuracy_goal = info['initial_accuracy_goal']
        gpu_setting = info['gpu_setting']
        
        for attack_method, method_abbr in attack_method_abbr.items():
            for gamma in gamma_values_stage_1:
                formatted_gamma = format_value(gamma)
                config = base_config.copy()
                config['training']['save_path'] = f"/tmp2/pfe/pattern/eyeg/{method_abbr}/{short_name}_g_{formatted_gamma}"
                config['training']['gpu_setting'] = gpu_setting
                config['model']['model_path'] = model_path
                config['attack']['method'] = attack_method
                config['attack']['gamma_adjustment'] = 'constant'
                config['attack']['gamma'] = gamma
                config['attack']['gamma_adjust_factor'] = 0.01
                config['attack']['accuracy_goal'] = initial_accuracy_goal
                config['dataset']['task_name'] = attribute
                config['dataset']['selected_attrs'] = [attribute]
                
                directory = f"eyeglasses/{method_abbr}"
                file_name = f"{short_name}_g_{formatted_gamma}.yml"
                save_config_to_yaml(config, directory, file_name)
                config_counter += 1

    print(f"Generated {config_counter} configuration files for Stage 1.")

gamma_results_stage_1 = {
    ('age', 'fairness constraint'): 5,
    ('age', 'perturbed fairness constraint'): 1,
    ('age', 'EquiMask fairness constraint'): 0.1,
    ('age', 'EquiMask perturbed fairness constraint'): 0.1,
}

# Stage 2: Find the Accuracy target
def generate_configs_stage_2(gamma_results):
    config_counter = 0
    accuracy_results = {}
    for attribute, info in attributes_info.items():
        short_name = info['short_name']
        model_path = info['model_path']
        accuracy_goals_stage_2 = info['accuracy_goals_stage_2']
        gpu_setting = info['gpu_setting']
        
        for attack_method, method_abbr in attack_method_abbr.items():
            gamma = gamma_results.get((attribute, attack_method), gamma_values_stage_1[0])
            
            for accuracy_goal in accuracy_goals_stage_2:
                formatted_accuracy_goal = format_value(accuracy_goal)
                config = base_config.copy()
                config['training']['save_path'] = f"/tmp2/pfe/pattern/eyeg/{method_abbr}/{short_name}_at_{formatted_accuracy_goal}"
                config['training']['gpu_setting'] = gpu_setting
                config['model']['model_path'] = model_path
                config['attack']['method'] = attack_method
                config['attack']['gamma_adjustment'] = 'dynamic'
                config['attack']['gamma'] = gamma
                config['attack']['gamma_adjust_factor'] = 0.01
                config['attack']['accuracy_goal'] = accuracy_goal
                config['dataset']['task_name'] = attribute
                config['dataset']['selected_attrs'] = [attribute]
                
                directory = f"eyeglasses/{method_abbr}"
                file_name = f"{short_name}_at_{formatted_accuracy_goal}.yml"
                save_config_to_yaml(config, directory, file_name)
                config_counter += 1

    print(f"Generated {config_counter} configuration files for Stage 2.")

accuracy_results_stage_2 = {
    ('age', 'fairness constraint'): 0.76,
    ('age', 'perturbed fairness constraint'): 0.8,
    ('age', 'EquiMask fairness constraint'): 0.74,
    ('age', 'EquiMask perturbed fairness constraint'): 0.76,
}

# Stage 3: Find the proper Gamma adjustment rate
def generate_configs_stage_3(gamma_results, accuracy_results):
    config_counter = 0
    for attribute, info in attributes_info.items():
        short_name = info['short_name']
        model_path = info['model_path']
        gpu_setting = info['gpu_setting']
        
        for attack_method, method_abbr in attack_method_abbr.items():
            gamma = gamma_results.get((attribute, attack_method), gamma_values_stage_1[0])
            accuracy_goal = accuracy_results.get((attribute, attack_method), info['accuracy_goals_stage_2'][0])
            
            for gamma_adjust_factor in gamma_adjust_factors_stage_3:
                if attack_method == 'EquiMask fairness constraint':
                    gamma = 0  # Special case for EquiMask
                
                formatted_gamma = format_value(gamma)
                formatted_accuracy_goal = format_value(accuracy_goal)
                formatted_gamma_adjust_factor = format_value(gamma_adjust_factor)
                config = base_config.copy()
                config['training']['save_path'] = f"/tmp2/pfe/pattern/eyeg/{method_abbr}/{short_name}_af_{formatted_gamma_adjust_factor}"
                config['training']['gpu_setting'] = gpu_setting
                config['model']['model_path'] = model_path
                config['attack']['method'] = attack_method
                config['attack']['gamma_adjustment'] = 'dynamic'
                config['attack']['gamma'] = gamma
                config['attack']['gamma_adjust_factor'] = gamma_adjust_factor
                config['attack']['accuracy_goal'] = accuracy_goal
                config['dataset']['task_name'] = attribute
                config['dataset']['selected_attrs'] = [attribute]
                
                directory = f"eyeglasses/{method_abbr}"
                file_name = f"{short_name}_af_{formatted_gamma_adjust_factor}.yml"
                save_config_to_yaml(config, directory, file_name)
                config_counter += 1

    print(f"Generated {config_counter} configuration files for Stage 3.")

# Run the configuration generation by stages
gamma_results_stage_1 = generate_configs_stage_1()
# accuracy_results_stage_2 = generate_configs_stage_2(gamma_results_stage_1)
# generate_configs_stage_3(gamma_results_stage_1, accuracy_results_stage_2)