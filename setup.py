import os
import torch
import numpy as np
import argparse
import yaml
import json
from datetime import datetime

def create_default_config():
    """Create default configuration files."""
    # Create config directory
    os.makedirs('config', exist_ok=True)
    
    # Default configuration
    default_config = {
        'data': {
            'root': '/path/to/rxr_dataset',
            'num_workers': 4,
            'max_length': 80
        },
        'model': {
            'embed_dim': 512,
            'num_heads': 8,
            'dropout': 0.1,
            'use_pose_traces': True,
            'use_ground_truth_path': True
        },
        'training': {
            'batch_size': 8,
            'num_epochs': 30,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'clip_model': 'ViT-B/32'
        }
    }
    
    # Write default config
    with open('config/default.yaml', 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    # Paths configuration
    paths_config = {
        'data_root': '/path/to/rxr_dataset',
        'output_dir': 'outputs',
        'checkpoint_dir': 'outputs/checkpoints',
        'visualization_dir': 'outputs/visualizations'
    }
    
    # Write paths config
    with open('config/paths.yaml', 'w') as f:
        yaml.dump(paths_config, f, default_flow_style=False)
    
    print("Created default configuration files in 'config' directory")

def create_init_files():
    """Create __init__.py files for all directories."""
    directories = [
        'src',
        'src/data',
        'src/models',
        'src/utils'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        init_file = os.path.join(directory, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                pass
    
    print("Created __init__.py files for all directories")

def create_requirements_file():
    """Create requirements.txt file."""
    requirements = [
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'numpy>=1.20.0',
        'matplotlib>=3.4.0',
        'Pillow>=8.2.0',
        'tqdm>=4.61.0',
        'pyyaml>=5.4.1',
        'scikit-learn>=0.24.2',
        'ftfy>=6.0.3',
        'regex>=2021.4.4',
        'clip @ git+https://github.com/openai/CLIP.git'
    ]
    
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))
    
    print("Created requirements.txt file")

def main():
    """Main function to set up the project."""
    parser = argparse.ArgumentParser(description='Set up RxR Heatmap project')
    parser.add_argument('--create_config', action='store_true', help='Create default configuration files')
    parser.add_argument('--create_init', action='store_true', help='Create __init__.py files')
    parser.add_argument('--create_requirements', action='store_true', help='Create requirements.txt file')
    parser.add_argument('--all', action='store_true', help='Perform all setup actions')
    
    args = parser.parse_args()
    
    if args.all or args.create_config:
        create_default_config()
    
    if args.all or args.create_init:
        create_init_files()
    
    if args.all or args.create_requirements:
        create_requirements_file()
    
    if not any([args.create_config, args.create_init, args.create_requirements, args.all]):
        parser.print_help()

if __name__ == "__main__":
    main()
