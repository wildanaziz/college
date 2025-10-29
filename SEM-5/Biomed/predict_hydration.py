import serial
import numpy as np
import joblib
from scipy.stats import entropy
import time

# must exist with training script for feature extraction
def extract_features(window_data):
    """Mengekstrak fitur statistik dari window data sensor."""
    if len(window_data) == 0:
        return None
        
    # calculate entropy in outside
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
    
    # make features dict
    features_dict = {
        'mean': np.mean(window_data),
        'variance': np.var(window_data),
        'std_dev': np.std(window_data),
        'entropy': calculated_entropy,
        'percentile_50': np.percentile(window_data, 50)
    }

    return np.array(list(features_dict.values()))


# WINDOWS SERIAL PORT
# SERIAL_PORT = 'COM3'

# LINUX SERIAL PORT
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600

# 60 x 10 Hz = 600 samples per window
SAMPLES_PER_WINDOW = 600

# load model and scaler
try:
    model = joblib.load('hydration_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    print("Error: File model 'hydration_model.pkl' atau 'scaler.pkl' tidak ditemukan.")
    print("Pastikan Anda sudah menjalankan 'train_model.py' terlebih dahulu.")
    exit()

# connect to arduino serial
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Terhubung ke Arduino di port {SERIAL_PORT}...")
    time.sleep(2)
except serial.SerialException as e:
    print(f"Error: Tidak dapat membuka port {SERIAL_PORT}. {e}")
    print("Pastikan port sudah benar dan Arduino terhubung.")
    exit()

gsr_window = []

print("Mulai mengumpulkan data untuk prediksi...")
print("Dibutuhkan 60 detik untuk prediksi pertama.")

try:
    while True:

        # read line from serial
        line = ser.readline().decode('utf-8').strip()
        
        if line:
            try:
                gsr_value = int(line)
                gsr_window.append(gsr_value)
                
                
                print(f"Mengumpulkan data... {len(gsr_window)}/{SAMPLES_PER_WINDOW}", end='\r')
                
                
                if len(gsr_window) >= SAMPLES_PER_WINDOW:
                    print("\nWindow 60 detik penuh. Mengekstrak fitur dan memprediksi...")
                    
                    # take last 600 samples
                    current_window_data = gsr_window[-SAMPLES_PER_WINDOW:] 
                    features = extract_features(current_window_data)
                    
                    # scale features
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    
                    # predict
                    prediction = model.predict(features_scaled)
                    
                    print("="*30)
                    print(f"   PREDIKSI STATUS: {prediction[0].upper()}")
                    print("="*30)
                    
                    # reset window
                    gsr_window = [] 
                    print("Memulai window 60 detik berikutnya...")

            except ValueError:
                print("Menerima data tidak valid, mengabaikan...")
                pass
                
except KeyboardInterrupt:
    print("\nProgram dihentikan.")
finally:
    ser.close()
    print("Koneksi serial ditutup.")