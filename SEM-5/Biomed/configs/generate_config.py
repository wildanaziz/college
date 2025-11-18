#!/usr/bin/env python3
"""
Quick configuration file generator for multiple experiments.
Usage: python generate_configs.py [dataset_file.csv]
Example: python generate_configs.py data_gsr_ripan_duduk_combine.csv
"""

import yaml
from pathlib import Path
import sys
import argparse

# Base configuration template
BASE_CONFIG = {
    'wandb': {
        'project': 'gsr-hydration-classification',
        'entity': None,
        'tags': [],
        'notes': ''
    },
    'dataset': {
        'file_path': 'data_gsr_ripan_duduk_combine.csv',
        'posture': 'sitting',
        'target_dir': 'data/combine'
    },
    'feature_extraction': {
        'window_size': '30S',
        'samples_per_window': 300,
        'overlap': 0
    },
    'model': {
        'type': 'KNN',
        'params': {
            'n_neighbors': 5,
            'weights': 'uniform',
            'metric': 'minkowski',
            'p': 2,
            'algorithm': 'auto'
        }
    },
    'training': {
        'test_size': 0.3,
        'random_state': 42,
        'stratify': True,
        'cross_validation': {
            'enabled': False,
            'cv_folds': 5
        }
    },
    'grid_search': {
        'enabled': False,
        'param_grid': {
            'n_neighbors': [3, 5, 7, 10, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'cv_folds': 5,
        'scoring': 'accuracy'
    },
    'model_saving': {
        'save_local': True,
        'save_wandb': True,
        'model_dir': 'model/models',
        'scaler_dir': 'model/scalers',
        'model_filename': 'knn_model.pkl',
        'scaler_filename': 'knn_scaler.pkl'
    },
    'logging': {
        'log_feature_stats': True,
        'log_confusion_matrix': True,
        'log_classification_report': True,
        'log_class_distribution': True,
        'verbose': True
    }
}

def get_dataset_info(dataset_path):
    """Extract person name and posture from dataset filename for config naming."""
    # Extract filename without extension
    filename = Path(dataset_path).stem
    
    # Example patterns:
    # "data_gsr_ripan_duduk_combine.csv" -> person: "ripan", posture: "duduk"
    # "data_gsr_john_standing_combine.csv" -> person: "john", posture: "standing"
    
    parts = filename.lower().split('_')
    
    # Common words to skip
    common_words = ['data', 'gsr', 'combine', 'dataset']
    
    # Posture keywords
    posture_keywords = {
        'duduk': 'duduk', 
        'berdiri': 'berdiri', 
        'sitting': 'sitting', 
        'standing': 'standing',
        'walk': 'walking',
        'walking': 'walking',
        'run': 'running',
        'running': 'running'
    }
    
    person_name = None
    posture = None
    
    # Extract person name and posture
    for i, part in enumerate(parts):
        # Skip common words
        if part in common_words:
            continue
        
        # Check if it's a posture keyword
        if part in posture_keywords:
            posture = part
            # Person name is usually before posture
            if person_name is None and i > 0:
                # Look backwards for person name
                for j in range(i-1, -1, -1):
                    if parts[j] not in common_words and parts[j] not in posture_keywords:
                        person_name = parts[j]
                        break
        else:
            # If not a posture and not common word, might be person name
            if person_name is None:
                person_name = part
    
    # Create prefix: person_posture or just person or just posture
    if person_name and posture:
        prefix = f"{person_name}_{posture}"
    elif person_name:
        prefix = person_name
    elif posture:
        prefix = posture
    else:
        # Fallback: use all meaningful parts
        meaningful_parts = [p for p in parts if p not in common_words]
        prefix = '_'.join(meaningful_parts[:2]) if meaningful_parts else 'default'
    
    return {
        'prefix': prefix,
        'person': person_name,
        'posture': posture
    }

def save_config(config, filename, target_dir='configs', dataset_prefix='', window_size='30S'):
    """Save configuration to YAML file in specified directory with dataset prefix and window size."""
    # Create target directory if it doesn't exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # Add dataset prefix and window size to filename if provided
    if dataset_prefix:
        # Insert prefix and window size before the extension
        name_parts = filename.rsplit('.', 1)
        if len(name_parts) == 2:
            filename = f"{name_parts[0]}_{dataset_prefix}_{window_size}.{name_parts[1]}"
        else:
            filename = f"{filename}_{dataset_prefix}_{window_size}"
    
    # Full path with directory
    filepath = Path(target_dir) / filename
    
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Created: {filepath}")

def generate_k_value_experiments(dataset_filename, dataset_info, target_data_dir, window_size='30S'):
    """Generate configs for different k values."""
    print("\nGenerating K-value experiments...")
    prefix = dataset_info['prefix']
    person = dataset_info['person']
    posture = dataset_info['posture']
    
    target_dir = f'configs/{prefix}/k_values'
    k_values = [3, 5, 7, 10, 15, 20]
    
    for k in k_values:
        config = BASE_CONFIG.copy()
        config = yaml.safe_load(yaml.dump(config))  # Deep copy
        
        # Set dataset filename only (not full path)
        config['dataset']['file_path'] = dataset_filename
        config['dataset']['target_dir'] = target_data_dir
        if posture:
            config['dataset']['posture'] = posture
        
        # Set window size
        config['feature_extraction']['window_size'] = window_size
        
        config['model']['params']['n_neighbors'] = k
        config['wandb']['tags'] = ['knn', person or 'unknown', posture or 'unknown', f'k{k}', window_size]
        config['wandb']['notes'] = f'KNN with k={k} on {prefix} dataset (window: {window_size})'
        config['model_saving']['model_filename'] = f'knn_{prefix}_k{k}_{window_size}.pkl'
        config['model_saving']['scaler_filename'] = f'scaler_{prefix}_k{k}_{window_size}.pkl'
        
        filename = f'config_k{k}.yaml'
        save_config(config, filename, target_dir, prefix, window_size)

def generate_distance_metric_experiments(dataset_filename, dataset_info, target_data_dir, window_size='30S'):
    """Generate configs for different distance metrics."""
    print("\nGenerating distance metric experiments...")
    prefix = dataset_info['prefix']
    person = dataset_info['person']
    posture = dataset_info['posture']
    
    target_dir = f'configs/{prefix}/distance_metrics'
    metrics = {
        'euclidean': {'metric': 'euclidean', 'p': 2},
        'manhattan': {'metric': 'manhattan', 'p': 1},
        'minkowski_p2': {'metric': 'minkowski', 'p': 2},
        'minkowski_p3': {'metric': 'minkowski', 'p': 3}
    }
    
    for name, params in metrics.items():
        config = BASE_CONFIG.copy()
        config = yaml.safe_load(yaml.dump(config))
        
        # Set dataset filename only (not full path)
        config['dataset']['file_path'] = dataset_filename
        config['dataset']['target_dir'] = target_data_dir
        if posture:
            config['dataset']['posture'] = posture
        
        # Set window size
        config['feature_extraction']['window_size'] = window_size
        
        config['model']['params']['metric'] = params['metric']
        config['model']['params']['p'] = params['p']
        config['wandb']['tags'] = ['knn', person or 'unknown', posture or 'unknown', name, window_size]
        config['wandb']['notes'] = f'Testing {name} distance metric on {prefix} (window: {window_size})'
        config['model_saving']['model_filename'] = f'knn_{prefix}_{name}_{window_size}.pkl'
        config['model_saving']['scaler_filename'] = f'scaler_{prefix}_{name}_{window_size}.pkl'
        
        filename = f'config_{name}.yaml'
        save_config(config, filename, target_dir, prefix, window_size)

def generate_weight_experiments(dataset_filename, dataset_info, target_data_dir, window_size='30S'):
    """Generate configs for different weighting schemes."""
    print("\nGenerating weight experiments...")
    prefix = dataset_info['prefix']
    person = dataset_info['person']
    posture = dataset_info['posture']
    
    target_dir = f'configs/{prefix}/weights'
    weights = ['uniform', 'distance']
    
    for weight in weights:
        config = BASE_CONFIG.copy()
        config = yaml.safe_load(yaml.dump(config))
        
        # Set dataset filename only (not full path)
        config['dataset']['file_path'] = dataset_filename
        config['dataset']['target_dir'] = target_data_dir
        if posture:
            config['dataset']['posture'] = posture
        
        # Set window size
        config['feature_extraction']['window_size'] = window_size
        
        config['model']['params']['weights'] = weight
        config['wandb']['tags'] = ['knn', person or 'unknown', posture or 'unknown', f'weight_{weight}', window_size]
        config['wandb']['notes'] = f'Testing {weight} weighting on {prefix} (window: {window_size})'
        config['model_saving']['model_filename'] = f'knn_{prefix}_weight_{weight}_{window_size}.pkl'
        config['model_saving']['scaler_filename'] = f'scaler_{prefix}_weight_{weight}_{window_size}.pkl'
        
        filename = f'config_weight_{weight}.yaml'
        save_config(config, filename, target_dir, prefix, window_size)

def generate_overlap_experiments(dataset_filename, dataset_info, target_data_dir, window_size='30S'):
    """Generate configs for different window overlaps."""
    print("\nGenerating window overlap experiments...")
    prefix = dataset_info['prefix']
    person = dataset_info['person']
    posture = dataset_info['posture']
    
    target_dir = f'configs/{prefix}/overlaps'
    overlaps = {
        'no_overlap': 0,
        'overlap_25': 75,  # 25% overlap
        'overlap_50': 150,  # 50% overlap
        'overlap_75': 225   # 75% overlap
    }
    
    for name, overlap in overlaps.items():
        config = BASE_CONFIG.copy()
        config = yaml.safe_load(yaml.dump(config))
        
        # Set dataset filename only (not full path)
        config['dataset']['file_path'] = dataset_filename
        config['dataset']['target_dir'] = target_data_dir
        if posture:
            config['dataset']['posture'] = posture
        
        # Set window size
        config['feature_extraction']['window_size'] = window_size
        config['feature_extraction']['overlap'] = overlap
        
        config['wandb']['tags'] = ['knn', person or 'unknown', posture or 'unknown', name, window_size]
        config['wandb']['notes'] = f'Testing {overlap} samples overlap ({name}) on {prefix} (window: {window_size})'
        config['model_saving']['model_filename'] = f'knn_{prefix}_{name}_{window_size}.pkl'
        config['model_saving']['scaler_filename'] = f'scaler_{prefix}_{name}_{window_size}.pkl'
        
        filename = f'config_{name}.yaml'
        save_config(config, filename, target_dir, prefix, window_size)

def generate_posture_experiments(dataset_filename, dataset_info, target_data_dir, window_size='30S'):
    """Generate configs for different postures."""
    print("\nGenerating posture-specific experiment...")
    prefix = dataset_info['prefix']
    person = dataset_info['person']
    posture = dataset_info['posture']
    
    target_dir = f'configs/{prefix}'
    
    # Determine parameters based on posture
    if posture in ['duduk', 'sitting']:
        k = 15
        metric = 'minkowski'
        posture_type = 'duduk' if posture == 'duduk' else 'sitting'
    elif posture in ['berdiri', 'standing']:
        k = 10
        metric = 'manhattan'
        posture_type = 'berdiri' if posture == 'berdiri' else 'standing'
    else:
        # Default parameters
        k = 5
        metric = 'minkowski'
        posture_type = posture or 'unknown'
    
    config = BASE_CONFIG.copy()
    config = yaml.safe_load(yaml.dump(config))
    
    # Set dataset filename only (not full path)
    config['dataset']['file_path'] = dataset_filename
    config['dataset']['target_dir'] = target_data_dir
    config['dataset']['posture'] = posture_type
    
    # Set window size
    config['feature_extraction']['window_size'] = window_size
    
    config['model']['params']['n_neighbors'] = k
    config['model']['params']['metric'] = metric
    config['wandb']['tags'] = ['knn', person or 'unknown', posture_type, 'journal_params', window_size]
    config['wandb']['notes'] = f'Journal paper parameters for {prefix} dataset (window: {window_size})'
    config['model_saving']['model_filename'] = f'knn_{prefix}_journal_{window_size}.pkl'
    config['model_saving']['scaler_filename'] = f'scaler_{prefix}_journal_{window_size}.pkl'
    
    filename = f'config_{prefix}_journal.yaml'
    save_config(config, filename, target_dir, '', window_size)

def generate_grid_search_config(dataset_filename, dataset_info, target_data_dir, window_size='30S'):
    """Generate comprehensive grid search config."""
    print("\nGenerating grid search config...")
    prefix = dataset_info['prefix']
    person = dataset_info['person']
    posture = dataset_info['posture']
    
    target_dir = f'configs/{prefix}'
    
    config = BASE_CONFIG.copy()
    config = yaml.safe_load(yaml.dump(config))
    
    # Set dataset filename only (not full path)
    config['dataset']['file_path'] = dataset_filename
    config['dataset']['target_dir'] = target_data_dir
    if posture:
        config['dataset']['posture'] = posture
    
    # Set window size
    config['feature_extraction']['window_size'] = window_size
    
    config['grid_search']['enabled'] = True
    config['grid_search']['param_grid'] = {
        'n_neighbors': [3, 5, 7, 10, 15, 20],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    config['wandb']['tags'] = ['knn', person or 'unknown', posture or 'unknown', 'grid_search', 'hyperparameter_tuning', window_size]
    config['wandb']['notes'] = f'Comprehensive grid search for optimal hyperparameters on {prefix} (window: {window_size})'
    config['model_saving']['model_filename'] = f'knn_{prefix}_best_params_{window_size}.pkl'
    config['model_saving']['scaler_filename'] = f'scaler_{prefix}_best_params_{window_size}.pkl'
    
    save_config(config, 'config_grid_search.yaml', target_dir, prefix, window_size)

def main():
    parser = argparse.ArgumentParser(
        description='Generate configuration files for GSR hydration classification experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate configs for Ripan sitting dataset with 30S window
  python generate_configs.py data_gsr_ripan_duduk_combine.csv
  
  # Generate configs for Ripan standing dataset with 60S window
  python generate_configs.py data_gsr_ripan_berdiri_combine.csv --window-size 60S
  
  # Generate configs for John sitting dataset with 45S window
  python generate_configs.py data_gsr_john_sitting_combine.csv --window-size 45S
  
  # Generate configs with custom data directory and window size
  python generate_configs.py my_data.csv --target-dir custom/path --window-size 30S
        '''
    )
    parser.add_argument(
        'dataset_file',
        nargs='?',
        default='data_gsr_ripan_duduk_combine.csv',
        help='Dataset filename (default: data_gsr_ripan_duduk_combine.csv)'
    )
    parser.add_argument(
        '--target-dir',
        default='data/combine',
        help='Target directory for dataset (default: data/combine)'
    )
    parser.add_argument(
        '--window-size',
        default='30S',
        help='Window size for feature extraction (default: 30S, options: 30S, 60S, etc.)'
    )
    
    args = parser.parse_args()
    
    # Get just the filename (not the full path)
    dataset_filename = Path(args.dataset_file).name
    window_size = args.window_size
    
    # Get dataset information
    dataset_info = get_dataset_info(dataset_filename)
    prefix = dataset_info['prefix']
    person = dataset_info['person']
    posture = dataset_info['posture']
    
    print("GSR HYDRATION CLASSIFICATION - CONFIG GENERATOR")
    print("="*60)
    print(f"\nDataset: {dataset_filename}")
    print(f"Person: {person or 'Not detected'}")
    print(f"Posture: {posture or 'Not detected'}")
    print(f"Prefix: {prefix}")
    print(f"Target directory: {args.target_dir}")
    print(f"Window size: {window_size}")
    
    # Update BASE_CONFIG with target directory and window size
    BASE_CONFIG['dataset']['target_dir'] = args.target_dir
    BASE_CONFIG['feature_extraction']['window_size'] = window_size
    
    print("\nGenerating configuration files...")
    
    # Generate different experiment types
    generate_k_value_experiments(dataset_filename, dataset_info, args.target_dir, window_size)
    generate_distance_metric_experiments(dataset_filename, dataset_info, args.target_dir, window_size)
    generate_weight_experiments(dataset_filename, dataset_info, args.target_dir, window_size)
    generate_overlap_experiments(dataset_filename, dataset_info, args.target_dir, window_size)
    generate_posture_experiments(dataset_filename, dataset_info, args.target_dir, window_size)
    generate_grid_search_config(dataset_filename, dataset_info, args.target_dir, window_size)
    
    print("\n" + "="*60)
    print("Configuration generation complete!")
    print("="*60)
    print(f"\nGenerated config files for: {person or 'unknown'} - {posture or 'unknown'} - {window_size}")
    print(f"\nconfigs/{prefix}/k_values/")
    print(f"   - config_k3_{prefix}_{window_size}.yaml to config_k20_{prefix}_{window_size}.yaml")
    print(f"\nconfigs/{prefix}/distance_metrics/")
    print(f"   - config_euclidean_{prefix}_{window_size}.yaml, config_manhattan_{prefix}_{window_size}.yaml, etc.")
    print(f"\nconfigs/{prefix}/weights/")
    print(f"   - config_weight_uniform_{prefix}_{window_size}.yaml, config_weight_distance_{prefix}_{window_size}.yaml")
    print(f"\nconfigs/{prefix}/overlaps/")
    print(f"   - config_no_overlap_{prefix}_{window_size}.yaml to config_overlap_75_{prefix}_{window_size}.yaml")
    print(f"\nconfigs/{prefix}/")
    print(f"   - config_{prefix}_journal_{window_size}.yaml")
    print(f"   - config_grid_search_{prefix}_{window_size}.yaml")
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    print("\nSingle experiment:")
    print(f"  python train_journal.py configs/{prefix}/k_values/config_k5_{prefix}_{window_size}.yaml")
    print("\nMultiple experiments (all k values):")
    print(f"  python run_experiments.py configs/{prefix}/k_values/config_k*_{prefix}_{window_size}.yaml")
    print("\nGrid search:")
    print(f"  python train_journal.py configs/{prefix}/config_grid_search_{prefix}_{window_size}.yaml")
    print("\nJournal baseline:")
    print(f"  python train_journal.py configs/{prefix}/config_{prefix}_journal_{window_size}.yaml")
    
    print("\n" + "="*60)
    print("DIRECTORY STRUCTURE")
    print("="*60)
    print(f"configs/{prefix}/")
    print(f"├── k_values/          (6 configs) - config_k*_{prefix}_{window_size}.yaml")
    print(f"├── distance_metrics/  (4 configs) - config_*_{prefix}_{window_size}.yaml")
    print(f"├── weights/           (2 configs) - config_weight_*_{prefix}_{window_size}.yaml")
    print(f"├── overlaps/          (4 configs) - config_*_overlap_{prefix}_{window_size}.yaml")
    print(f"├── config_{prefix}_journal_{window_size}.yaml")
    print(f"└── config_grid_search_{prefix}_{window_size}.yaml")
    
    print("\n" + "="*60)
    print("TIP: Generate configs for other people/postures/window sizes:")
    print("="*60)
    print("  python generate_configs.py data_gsr_ripan_berdiri_combine.csv --window-size 30S")
    print("  python generate_configs.py data_gsr_ripan_berdiri_combine.csv --window-size 60S")
    print("  python generate_configs.py data_gsr_john_sitting_combine.csv --window-size 45S")
    print("  python generate_configs.py data_gsr_alice_walking_combine.csv --window-size 30S")
    print()

if __name__ == "__main__":
    main()