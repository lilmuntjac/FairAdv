import os
import yaml

# Define the base configuration dictionary
base_config = {
    'training': {
        'batch_size': 128,
        'final_epoch': 2,
        'random_seed': 2665,
        'scheduler': None,
        'use_cuda': True,
        'gpu_setting': ''
    },
    'reweight': {
        'iteration': 10,
        'eta': ''
    },
    'dataset': {
        'task_name': '',
        'training_schema': 'reweight',
        'name': '',
        'type': 'binary',
        'balanced': False,
        'selected_attrs': [],
        'protected_attr': '',
        'num_outputs': 1
    }
}

# Attributes info with model paths and accuracy goals for different stages
attributes_info = {
    # 'Attractive': {
    #     'dataset_name': 'celeba',
    #     'protected_attr': 'Male',
    #     'short_name': 'cb_at',
    #     'gpu_setting': '0'
    # },
    'Big_Nose': {
        'dataset_name': 'celeba',
        'protected_attr': 'Male',
        'short_name': 'cb_bn',
        'gpu_setting': '0'
    },
    'Bags_Under_Eyes': {
        'dataset_name': 'celeba',
        'protected_attr': 'Male',
        'short_name': 'cb_bu',
        'gpu_setting': '0'
    },
    'High_Cheekbones': {
        'dataset_name': 'celeba',
        'protected_attr': 'Male',
        'short_name': 'cb_hc',
        'gpu_setting': '0'
    },
    'Oval_Face': {
        'dataset_name': 'celeba',
        'protected_attr': 'Male',
        'short_name': 'cb_of',
        'gpu_setting': '0'
    },
    'Young': {
        'dataset_name': 'celeba',
        'protected_attr': 'Male',
        'short_name': 'cb_yo',
        'gpu_setting': '0'
    },
    'age': {
        'dataset_name': 'fairface',
        'protected_attr': 'race',
        'short_name': 'ff',
        'gpu_setting': '0'
    }
}

# Define variations for specific fields
eta_value = [0.01, 0.1, 1, 10, 100, 1000]
epoch_value = [2, 4, 6]


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

def generate_configs():
    config_counter = 0
    for attribute, info in attributes_info.items():
        dataset_name = info['dataset_name']
        protected_attr = info['protected_attr']
        short_name = info['short_name']
        gpu_setting = info['gpu_setting']

        max_length = max(len(f"{value:.2f}".replace('.', 'd')) for value in eta_value)
        for eta in eta_value:
            for epoch in epoch_value:
                formatted_eta = format_value(eta, max_length)
                config = base_config.copy()
                config['training']['save_path'] = f"/tmp2/pfe/lbc/{short_name}_eta_{formatted_eta}_e{epoch:02d}"
                config['training']['gpu_setting'] = gpu_setting
                config['reweight']['eta'] = eta
                config['dataset']['task_name'] = attribute
                config['dataset']['name'] = dataset_name
                config['dataset']['selected_attrs'] = [attribute]
                config['dataset']['protected_attr'] = protected_attr

                file_name = f"{short_name}_eta_{formatted_eta}_e{epoch:02d}.yml"
                save_config_to_yaml(config, '.', file_name)
                config_counter += 1

    print(f"Generated {config_counter} configuration files.")

# Run the configuration generation
generate_configs()
