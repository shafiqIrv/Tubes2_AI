import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from time import time

# Ngitung entropy 
def entropy(y):
    if len(y) == 0:
        return 0
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

# Menghitung information gain untuk sebuah fitur dan threshold tertentu
def information_gain(data, feature, target, min_samples=50):
    # Skip kalo datanya terlalu dikit
    if len(target) < min_samples:
        return 0, None
        
    total_entropy = entropy(target)
    
    # Inisialisasi threshold 
    best_threshold = None
    best_gain = -1

    # Cari nilai unik untuk split
    unique_values = np.unique(data[feature])
    
    # Kalo cuma ada 1 nilai unique, ga bisa di-split
    if len(unique_values) <= 1:
        return 0, unique_values[0] if len(unique_values) == 1 else None

    # Kalo nilai uniknya kebanyakan, ambil sebagian aja
    if len(unique_values) > 10:
        percentiles = np.percentile(data[feature], [25, 50, 75])
        unique_values = np.unique(np.concatenate([unique_values[:5], percentiles, unique_values[-5:]]))

    # Iterasi nilai threshold yang mungkin
    for threshold in unique_values:
        # Pake mask daripada bikin copy data
        left_mask = data[feature] <= threshold
        right_mask = ~left_mask
        
        # Skip kalo splitnya ga seimbang
        if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
            continue

        # Hitung entropy masing-masing split
        left_entropy = entropy(target[left_mask])
        right_entropy = entropy(target[right_mask])
        
        # Hitung bobot berdasarkan jumlah sampel
        weight_left = np.sum(left_mask) / len(target)
        weight_right = np.sum(right_mask) / len(target)
        
        # Hitung entropy setelah split
        split_entropy = weight_left * left_entropy + weight_right * right_entropy
        
        # Menghitung information gain
        gain = total_entropy - split_entropy

        # Memperbarui threshold terbaik jika gain lebih baik
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_gain, best_threshold


class ID3:
    def __init__(self, max_depth=None, min_samples=50, min_gain=1e-4):
        self.tree = None
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_gain = min_gain
        self.label_encoder = LabelEncoder()
        self.n_samples = 0
        self.start_time = None
        self.message = "ID3 Numeric Classifier"

    # Membangun decision tree secara rekursif
    def fitter(self, X, y, depth=0):
        n_samples = len(y)
        
        # Print progress tiap beberapa level
        if depth % 2 == 0:
            elapsed = time() - self.start_time
            progress = (self.n_samples - n_samples) / self.n_samples * 100
            print(f"Level {depth}, Progress: {progress:.1f}%, "
                  f"Samples: {n_samples}, Time: {elapsed:.1f}s")

        # Base cases - bikin leaf node
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples or \
           len(np.unique(y)) == 1:
            return np.bincount(y).argmax()

        # Inisialisasi pencarian split terbaik
        best_feature = None
        best_threshold = None
        best_gain = -1

        # Cari feature dan threshold terbaik
        for feature in X.columns:
            gain, threshold = information_gain(X, feature, y, self.min_samples)
            if gain > best_gain and gain > self.min_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

        # Kalo ga ketemu split yang bagus, jadiin leaf node
        if best_gain <= self.min_gain or best_threshold is None:
            return np.bincount(y).argmax()

        # Bikin split pake mask
        left_mask = X[best_feature] <= best_threshold
        right_mask = ~left_mask

        # Cek split identik
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return np.bincount(y).argmax()

        # Bikin node dan rekursi ke child nodes
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self.fitter(X[left_mask], y[left_mask], depth + 1),
            'right': self.fitter(X[right_mask], y[right_mask], depth + 1)
        }
    
    def fit(self, X, y):
        # Encode label kalo bukan numerik
        self.n_samples = len(y)
        self.start_time = time()
        print(f"Mulai training dengan {self.n_samples} samples...")
        
        # Transform label jadi numerik
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Build tree
        self.tree = self.fitter(X, y_encoded)
        print(f"Training selesai dalam {time() - self.start_time:.1f} detik")

    # Prediksi kelas untuk satu instance
    def predict_instance(self, instance, tree):
        if not isinstance(tree, dict):
            return tree

        if instance[tree['feature']] <= tree['threshold']:
            return self.predict_instance(instance, tree['left'])
        else:
            return self.predict_instance(instance, tree['right'])

    # Prediksi kelas untuk banyak instances
    def predict(self, X):
        numeric_predictions = [self.predict_instance(row, self.tree) 
                             for _, row in X.iterrows()]
        # Kembalikan ke label asli
        return self.label_encoder.inverse_transform(numeric_predictions)

    # Simpan model
    def save_model(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved to {file_name}")

    # Load model
    @staticmethod
    def load_model(file_name):
        with open(file_name, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {file_name}")
        return model