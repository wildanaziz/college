import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- Fungsi Ekstraksi Fitur ---
# Berdasarkan fitur terbaik dari jurnal (Kombinasi A, B, C)
# Kita akan gunakan gabungan fitur terbaik: Mean, Variance, Entropy, Percentile, Standard Deviation
def extract_features(window_data):
    """Mengekstrak fitur statistik dari window data sensor."""
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
    except ValueError as e:
        calculated_entropy = 0.0
    
    features = {
        'mean': np.mean(window_data),
        'variance': np.var(window_data),
        'std_dev': np.std(window_data),
        'entropy': calculated_entropy,
        'percentile_50': np.percentile(window_data, 50) # Median
    }
    return features

# prepare dataset
try:
    df = pd.read_csv('data_gsr.csv')
except FileNotFoundError:
    print("Error: File 'data_gsr.csv' tidak ditemukan.")
    print("Silakan kumpulkan data Anda terlebih dahulu.")
    exit()

# convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# determine window size and features
WINDOW_SIZE = '60S'

all_features = []
all_labels = []

# windowing and feature extraction
for label, group in df.groupby('label'):
    
    # 60 x 10 Hz = 600 samples per window
    samples_per_window = 600 
    
    gsr_values = group['gsr_value'].values
    
    for i in range(0, len(gsr_values) - samples_per_window + 1, samples_per_window):
        window = gsr_values[i : i + samples_per_window]
        
        # extract features from the window
        features = extract_features(window)
        
        if features:
            all_features.append(features)
            all_labels.append(label)

if not all_features:
    print("Error: Tidak ada fitur yang diekstrak. Periksa data Anda.")
    print("Pastikan Anda memiliki cukup data (minimal 60 detik per sampel).")
    exit()

# prepare data for model training
X = pd.DataFrame(all_features)
y = np.array(all_labels)

X = X.fillna(0).replace([np.inf, -np.inf], 0)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# MUST for knn - feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train
knn_model = KNeighborsClassifier(n_neighbors=15, metric='minkowski', weights='uniform') # [cite: 232, 268]
knn_model.fit(X_train_scaled, y_train)

y_pred = knn_model.predict(X_test_scaled)
print("--- Hasil Evaluasi Model ---")
print(f"Akurasi: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

joblib.dump(knn_model, 'hydration_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel dan scaler berhasil disimpan!")