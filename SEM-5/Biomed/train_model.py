import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import joblib
import warnings
import wandb
import yaml
import os
from pathlib import Path

warnings.filterwarnings('ignore')

#  Load Configuration from YAML 
def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[DONE] Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"[FAILED] Error: Config file not found: {config_path}")
        print("  Please create a config.yaml file or specify a valid config path.")
        exit()

# Parse command line arguments
import argparse

parser = argparse.ArgumentParser(description='Train KNN model for GSR hydration classification')
parser.add_argument('config', nargs='?', default='config.yaml', 
                    help='Path to configuration YAML file (default: config.yaml)')
args = parser.parse_args()

# Load config
cfg = load_config(args.config)

#  Initialize Weights & Biases 
run_name = f"{cfg['wandb']['tags'][1]}-{cfg['wandb']['tags'][3]}-{cfg['model']['type']}-{cfg['dataset']['posture']}-k{cfg['model']['params']['n_neighbors']}-{cfg['wandb']['tags'][4]}"
wandb.init(
    project=cfg['wandb']['project'],
    entity=cfg['wandb']['entity'],
    name=run_name,
    tags=cfg['wandb']['tags'],
    notes=cfg['wandb']['notes'],
    config={
        **cfg['dataset'],
        **cfg['feature_extraction'],
        **cfg['model'],
        **cfg['training']
    }
)

config = wandb.config

#  Feature Extraction Function 
def extract_features(window_data):
    """Extract statistical features from window data."""
    if len(window_data) == 0:
        return None
        
    try:
        window_data_int = np.round(window_data).astype(int)
        min_val = np.min(window_data_int)
        if min_val < 0:
            window_data_int = window_data_int - min_val
            
        counts = np.bincount(window_data_int)
        probabilities = counts / len(window_data_int)
        probabilities = probabilities[probabilities > 0] 
        calculated_entropy = entropy(probabilities)
    except ValueError:
        calculated_entropy = 0.0
    
    features = {
        'mean': np.mean(window_data),
        'variance': np.var(window_data),
        'std_dev': np.std(window_data),
        'entropy': calculated_entropy,
        'percentile_50': np.percentile(window_data, 50)
    }
    return features

#  Load Dataset 
def load_dataset(cfg):
    """Load and prepare dataset."""
    target_dir = cfg['dataset']['target_dir']
    file_path = cfg['dataset']['file_path']
    full_path = f'{target_dir}/{file_path}'
    
    try:
        df = pd.read_csv(full_path)
        print(f"[DONE] Dataset loaded: {file_path}")
        print(f"  Total samples: {len(df)}")
        
        if cfg['logging']['log_class_distribution']:
            class_dist = df['label'].value_counts().to_dict()
            print(f"  Class distribution: {class_dist}")
            wandb.log({
                "dataset_size": len(df),
                "class_distribution": class_dist
            })
        
        return df
    except FileNotFoundError:
        print(f"[FAILED] Error: File {full_path} not found.")
        wandb.finish()
        exit()

#  Extract Features from Dataset 
def extract_dataset_features(df, cfg):
    """Extract features from entire dataset."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    samples_per_window = cfg['feature_extraction']['samples_per_window']
    overlap = cfg['feature_extraction'].get('overlap', 0)
    step_size = samples_per_window - overlap
    
    all_features = []
    all_labels = []
    
    print("\n[DONE] Extracting features...")
    for label, group in df.groupby('label'):
        gsr_values = group['gsr_value'].values
        
        for i in range(0, len(gsr_values) - samples_per_window + 1, step_size):
            window = gsr_values[i : i + samples_per_window]
            features = extract_features(window)
            
            if features:
                all_features.append(features)
                all_labels.append(label)
    
    print(f"  Total windows extracted: {len(all_features)}")
    wandb.log({"total_windows_extracted": len(all_features)})
    
    if not all_features:
        print("[FAILED] Error: No features extracted. Check your data.")
        wandb.finish()
        exit()
    
    return all_features, all_labels

#  Train Model 
def train_model(X_train_scaled, y_train, cfg):
    """Train KNN model."""
    model_params = cfg['model']['params']
    
    if cfg['grid_search']['enabled']:
        print("\n[DONE] Running Grid Search for hyperparameter tuning...")
        param_grid = cfg['grid_search']['param_grid']
        
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(
            knn, 
            param_grid, 
            cv=cfg['grid_search']['cv_folds'],
            scoring=cfg['grid_search']['scoring'],
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")
        
        wandb.log({
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_
        })
        
        return grid_search.best_estimator_
    else:
        print("\n[DONE] Training KNN model...")
        knn_model = KNeighborsClassifier(**model_params)
        knn_model.fit(X_train_scaled, y_train)
        return knn_model

#  Evaluate Model 
def evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, cfg):
    """Evaluate model performance."""
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, average='weighted')
    test_recall = recall_score(y_test, y_pred_test, average='weighted')
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    
    print("MODEL EVALUATION RESULTS")
    print(f"Train Accuracy:  {train_acc:.4f}")
    print(f"Test Accuracy:   {test_acc:.4f}")
    print(f"Test Precision:  {test_precision:.4f}")
    print(f"Test Recall:     {test_recall:.4f}")
    print(f"Test F1-Score:   {test_f1:.4f}")
    
    if cfg['logging']['verbose']:
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred_test))
    
    # Log metrics to W&B
    wandb.log({
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1_score": test_f1
    })
    
    # Confusion Matrix
    if cfg['logging']['log_confusion_matrix']:
        # Get all unique classes from both train and test
        all_classes = sorted(list(set(list(y_train) + list(y_test))))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred_test, labels=all_classes)
        
        # Create DataFrame for better visualization
        cm_df = pd.DataFrame(
            cm,
            index=[f"True: {c}" for c in all_classes],
            columns=[f"Pred: {c}" for c in all_classes]
        )
        
        # Log as table (more reliable than plot)
        wandb.log({"confusion_matrix": wandb.Table(dataframe=cm_df)})
        
        # Also log as matplotlib figure for better visualization
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=all_classes, yticklabels=all_classes,
                       ax=ax, cbar_kws={'label': 'Count'})
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Confusion Matrix')
            plt.tight_layout()
            
            wandb.log({"confusion_matrix_plot": wandb.Image(fig)})
            plt.close(fig)
        except ImportError:
            pass  # matplotlib/seaborn not available
    
    # Classification Report
    if cfg['logging']['log_classification_report']:
        report = classification_report(y_test, y_pred_test, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        wandb.log({"classification_report": wandb.Table(dataframe=report_df)})
    
    return test_acc

#  Save Model 
def save_model(model, scaler, cfg):
    """Save model and scaler."""
    if not cfg['model_saving']['save_local'] and not cfg['model_saving']['save_wandb']:
        return
    
    model_dir = cfg['model_saving']['model_dir']
    scaler_dir = cfg['model_saving']['scaler_dir']
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)
    
    model_path = f"{model_dir}/{cfg['model_saving']['model_filename']}"
    scaler_path = f"{scaler_dir}/{cfg['model_saving']['scaler_filename']}"
    
    if cfg['model_saving']['save_local']:
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"\n[DONE] Model saved: {model_path}")
        print(f"[DONE] Scaler saved: {scaler_path}")
    
    if cfg['model_saving']['save_wandb']:
        artifact_model = wandb.Artifact('knn-model', type='model')
        artifact_model.add_file(model_path)
        wandb.log_artifact(artifact_model)
        
        artifact_scaler = wandb.Artifact('knn-scaler', type='preprocessor')
        artifact_scaler.add_file(scaler_path)
        wandb.log_artifact(artifact_scaler)
        
        print("[DONE] Model and scaler uploaded to W&B!")

#  Main Execution 
def main():
    print("GSR HYDRATION CLASSIFICATION - KNN MODEL")
    
    # Load dataset
    df = load_dataset(cfg)
    
    # Extract features
    all_features, all_labels = extract_dataset_features(df, cfg)
    
    # Prepare data
    X = pd.DataFrame(all_features)
    y = np.array(all_labels)
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    if cfg['logging']['log_feature_stats']:
        wandb.log({"feature_statistics": wandb.Table(dataframe=X.describe())})
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg['training']['test_size'],
        random_state=cfg['training']['random_state'],
        stratify=y if cfg['training']['stratify'] else None
    )
    
    print(f"\n[DONE] Data split: {len(X_train)} train, {len(X_test)} test")
    wandb.log({
        "train_size": len(X_train),
        "test_size": len(X_test)
    })
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = train_model(X_train_scaled, y_train, cfg)
    
    # Evaluate model
    test_acc = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, cfg)
    
    # Save model
    save_model(model, scaler, cfg)
    
    print("\n[DONE] Training completed successfully!")
    print(f"[DONE] Final Test Accuracy: {test_acc:.4f}")
    print(f"[DONE] View results at: {wandb.run.get_url()}")
    
    wandb.finish()

if __name__ == "__main__":
    main()