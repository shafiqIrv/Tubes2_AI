import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Ngitung entropy 
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

# Menghitung information gain untuk sebuah fitur dan threshold tertentu
def information_gain(data, feature, target):
    total_entropy = entropy(target)
    
    # Inisialisasi threshold 
    best_threshold = None
    best_gain = -1

    # Iterasi pada nilai unik dalam fitur untuk mencari threshold terbaik
    unique_values = np.unique(data[feature])
    for threshold in unique_values:
        # Membagi data menjadi dua grup berdasarkan threshold
        left_split = target[data[feature] <= threshold]
        right_split = target[data[feature] > threshold]
        
        # Menghitung entropy dari masing-masing grup
        weight_left = len(left_split) / len(target)
        weight_right = len(right_split) / len(target)
        split_entropy = (weight_left * entropy(left_split)) + (weight_right * entropy(right_split))
        
        # Menghitung information gain
        gain = total_entropy - split_entropy

        # Memperbarui threshold terbaik jika gain lebih baik
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_gain, best_threshold


class ID3Algorithm:
    def __init__(self, max_depth=None):
        self.tree = None
        self.max_depth = max_depth
        self.message = "ID3 Numeric Classifier"

    # Membangun decision tree secara rekursif
    def fitter(self, X, y, depth=0):
        # Jika semua data memiliki kelas yang sama/kedalaman maksimum tercapai, bikin node daun
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return np.unique(y)[0]  # Node daun dengan kelas dominan

        # Inisialisasi fitur, threshold, dan gain terbaik
        best_feature = None
        best_threshold = None
        best_gain = -1

        # Cari fitur dan threshold terbaik dari information gain
        for feature in X.columns:
            gain, threshold = information_gain(X, feature, y)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

        # Kalo gada gain, buat node daun dengan kelas dominan
        if best_gain == 0:
            return np.unique(y)[0]

        # Bikin node internal dengan fitur dan threshold terbgs
        tree = {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': None,
            'right': None
        }

        # Rekursif ke cabang kiri dan kanan
        left_indices = X[best_feature] <= best_threshold
        right_indices = X[best_feature] > best_threshold

        tree['left'] = self.fitter(X[left_indices], y[left_indices], depth + 1)
        tree['right'] = self.fitter(X[right_indices], y[right_indices], depth + 1)

        # self.tree = tree
        return tree
    
    def fit(self, X, y):
        self.tree = self.fitter(X, y)

    # Prediksi kelas untuk satu instance menggunakan decision tree
    def predict_instance(self, instance, tree):
        # Jika tree adalah node daun, kembalikan kelasnya
        if not isinstance(tree, dict):
            return tree

        # Arahkan ke cabang kiri atau kanan berdasarkan nilai threshold
        feature = tree['feature']
        threshold = tree['threshold']

        if instance[feature] <= threshold:
            return self.predict_instance(instance, tree['left'])
        else:
            return self.predict_instance(instance, tree['right'])

    # Prediksi kelas untuk semua data (hasilnya adalah list dengan kelasnnya)
    def predict(self, X):
        return [self.predict_instance(row, self.tree) for _, row in X.iterrows()]

    # Simpan model
    def save_model(self, file_name):
        """Save the model to a file."""
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved to {file_name}")

    # Load model
    @staticmethod
    def load_model(file_name):
        """Load the model from a file."""
        with open(file_name, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {file_name}")
        return model