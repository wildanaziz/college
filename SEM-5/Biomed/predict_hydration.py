#!/usr/bin/env python3
"""
Real-time GSR emotion/hydration prediction from Arduino serial data.
Usage: python predict_realtime.py [--model MODEL_PATH] [--scaler SCALER_PATH] [--config CONFIG_PATH]
"""

import serial
import numpy as np
import joblib
from scipy.stats import entropy
import time
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Feature extraction function (must match training script)
def extract_features(window_data):
    """Extract statistical features from sensor window data."""
    if len(window_data) == 0:
        return None
        
    # Calculate entropy
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
    
    # Create features dictionary (must match training order)
    features_dict = {
        'mean': np.mean(window_data),
        'variance': np.var(window_data),
        'std_dev': np.std(window_data),
        'entropy': calculated_entropy,
        'percentile_50': np.percentile(window_data, 50)
    }

    return np.array(list(features_dict.values()))


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[SUCCESS] Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"[FAILED] Config file not found: {config_path}")
        return None


def find_model_and_scaler(config=None, model_path=None, scaler_path=None):
    """
    Find model and scaler files from config or explicit paths.
    Priority: explicit paths > config paths > default paths
    """
    # If explicit paths provided, use them
    if model_path and scaler_path:
        return model_path, scaler_path
    
    # Try to get from config
    if config:
        model_dir = config.get('model_saving', {}).get('model_dir', 'model/models')
        scaler_dir = config.get('model_saving', {}).get('scaler_dir', 'model/scalers')
        model_filename = config.get('model_saving', {}).get('model_filename', 'knn_model.pkl')
        scaler_filename = config.get('model_saving', {}).get('scaler_filename', 'knn_scaler.pkl')
        
        model_path = f"{model_dir}/{model_filename}"
        scaler_path = f"{scaler_dir}/{scaler_filename}"
        
        return model_path, scaler_path
    
    # Default paths
    return 'model/models/knn_model.pkl', 'model/scalers/knn_scaler.pkl'


def load_model_and_scaler(model_path, scaler_path):
    """Load trained model and scaler."""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(f"[SUCCESS] Model loaded: {model_path}")
        print(f"[SUCCESS] Scaler loaded: {scaler_path}")
        return model, scaler
    except FileNotFoundError as e:
        print(f"[FAILED] Error: Model or scaler file not found")
        print(f"  Model: {model_path}")
        print(f"  Scaler: {scaler_path}")
        print("\nPlease train a model first using:")
        print("  python train_journal.py config.yaml")
        return None, None


def connect_serial(port, baud_rate):
    """Connect to Arduino via serial port."""
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
        print(f"[SUCCESS] Connected to Arduino on port {port}")
        time.sleep(2)  # Wait for Arduino to initialize
        return ser
    except serial.SerialException as e:
        print(f"[FAILED] Error: Cannot open port {port}")
        print(f"  {e}")
        print("\nTroubleshooting:")
        print("  - Check if Arduino is connected")
        print("  - Verify the correct port (COM3 on Windows, /dev/ttyUSB0 on Linux)")
        print("  - Try: ls /dev/tty* (Linux) or check Device Manager (Windows)")
        return None


def predict_realtime(ser, model, scaler, samples_per_window, log_predictions=False):
    """Run real-time prediction loop."""
    gsr_window = []
    prediction_count = 0
    start_time = time.time()
    
    # Log file setup
    log_file = None
    if log_predictions:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"predictions_log_{timestamp}.csv"
        with open(log_file, 'w') as f:
            f.write("timestamp,prediction,confidence,window_mean,window_std\n")
        print(f"[SUCCESS] Logging predictions to: {log_file}")
    
    window_duration = samples_per_window / 10  # Assuming 10 Hz sampling rate
    print(f"\n{'='*60}")
    print(f"REAL-TIME PREDICTION STARTED")
    print(f"{'='*60}")
    print(f"Window size: {samples_per_window} samples ({window_duration:.0f} seconds)")
    print(f"Collecting data for first prediction...")
    print(f"Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Read line from serial
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if line:
                try:
                    gsr_value = int(line)
                    gsr_window.append(gsr_value)
                    
                    # Progress indicator
                    progress = len(gsr_window) / samples_per_window * 100
                    print(f"Collecting data... {len(gsr_window)}/{samples_per_window} ({progress:.1f}%)", end='\r')
                    
                    # When window is full, make prediction
                    if len(gsr_window) >= samples_per_window:
                        print("\n" + "="*60)
                        print(f"Window complete. Extracting features and predicting...")
                        
                        # Take last N samples
                        current_window_data = gsr_window[-samples_per_window:]
                        features = extract_features(current_window_data)
                        
                        if features is None:
                            print("[FAILED] Feature extraction failed")
                            gsr_window = []
                            continue
                        
                        # Scale features
                        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                        features_scaled = scaler.transform(features.reshape(1, -1))
                        
                        # Predict
                        prediction = model.predict(features_scaled)[0]
                        
                        # Get prediction probability/confidence if available
                        confidence = None
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(features_scaled)[0]
                            confidence = np.max(proba)
                        
                        # Display prediction
                        prediction_count += 1
                        elapsed_time = time.time() - start_time
                        
                        print(f"\n{'*'*60}")
                        print(f"  PREDICTION #{prediction_count}")
                        print(f"  Status: {prediction.upper()}")
                        if confidence:
                            print(f"  Confidence: {confidence*100:.2f}%")
                        print(f"  Window mean: {np.mean(current_window_data):.2f}")
                        print(f"  Window std: {np.std(current_window_data):.2f}")
                        print(f"  Elapsed time: {elapsed_time:.1f}s")
                        print(f"{'*'*60}\n")
                        
                        # Log prediction
                        if log_predictions:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            conf_str = f"{confidence:.4f}" if confidence else "N/A"
                            with open(log_file, 'a') as f:
                                f.write(f"{timestamp},{prediction},{conf_str},{np.mean(current_window_data):.2f},{np.std(current_window_data):.2f}\n")
                        
                        # Reset window (or use sliding window)
                        gsr_window = []
                        print(f"Starting next window...")
                        
                except ValueError:
                    # Invalid data received, ignore
                    pass
                    
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("PREDICTION STOPPED")
        print("="*60)
        print(f"Total predictions: {prediction_count}")
        print(f"Total time: {time.time() - start_time:.1f}s")
        if log_predictions:
            print(f"Predictions saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time GSR emotion/hydration prediction from Arduino',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Use default model and scaler
  python predict_realtime.py
  
  # Specify model and scaler
  python predict_realtime.py --model model/models/knn_ripan_duduk_k5.pkl --scaler model/scalers/scaler_ripan_duduk_k5.pkl
  
  # Use config file to find model
  python predict_realtime.py --config configs/ripan_duduk/config_ripan_duduk_journal.yaml
  
  # Custom serial port
  python predict_realtime.py --port COM3
  
  # Enable logging
  python predict_realtime.py --log
        '''
    )
    
    parser.add_argument('--model', type=str, help='Path to model file (.pkl)')
    parser.add_argument('--scaler', type=str, help='Path to scaler file (.pkl)')
    parser.add_argument('--config', type=str, help='Path to config file (.yaml)')
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0', 
                        help='Serial port (default: /dev/ttyUSB0)')
    parser.add_argument('--baud', type=int, default=9600, 
                        help='Baud rate (default: 9600)')
    parser.add_argument('--samples', type=int, default=300,
                        help='Samples per window (default: 300)')
    parser.add_argument('--log', action='store_true',
                        help='Log predictions to CSV file')
    
    args = parser.parse_args()
    
    print("="*60)
    print("GSR REAL-TIME PREDICTION")
    print("="*60)
    print()
    
    # Load config if provided
    config = None
    if args.config:
        config = load_config(args.config)
        if config:
            # Update samples per window from config if available
            if 'feature_extraction' in config:
                args.samples = config['feature_extraction'].get('samples_per_window', args.samples)
    
    # Find model and scaler paths
    model_path, scaler_path = find_model_and_scaler(config, args.model, args.scaler)
    
    # Load model and scaler
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    if model is None or scaler is None:
        return
    
    # Display model info
    print(f"\nModel type: {type(model).__name__}")
    if hasattr(model, 'n_neighbors'):
        print(f"K neighbors: {model.n_neighbors}")
        print(f"Metric: {model.metric}")
        print(f"Weights: {model.weights}")
    
    # Connect to serial port
    print(f"\nConnecting to serial port...")
    ser = connect_serial(args.port, args.baud)
    if ser is None:
        return
    
    # Run prediction loop
    try:
        predict_realtime(ser, model, scaler, args.samples, args.log)
    finally:
        ser.close()
        print("\n[SUCCESS] Serial connection closed")


if __name__ == "__main__":
    main()