import yaml
import argparse

def update_gpu_setting(yml_file_path, new_gpu_setting):
    # Read the YAML file
    with open(yml_file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Update the gpu_setting in the training section
    if 'training' in data and 'gpu_setting' in data['training']:
        data['training']['gpu_setting'] = new_gpu_setting

    # Save the updated YAML back to the same file
    with open(yml_file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    print(f"Updated gpu_setting to {new_gpu_setting} in {yml_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update GPU setting in a YAML file.')
    parser.add_argument('yml_file_path', type=str, help='Path to the YAML file')
    parser.add_argument('new_gpu_setting', type=str, help='New GPU setting value')

    args = parser.parse_args()

    update_gpu_setting(args.yml_file_path, args.new_gpu_setting)
