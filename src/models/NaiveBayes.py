import numpy as np
import pandas as pd
from collections import defaultdict
import pickle

class GaussianNaiveBayes:
    def __init__(self):
        self.priors = {}
        self.mean_std = defaultdict(dict)
        self.classes = None 
    
    def fit(self, X, y):
        # Pastikan y berbentuk array numpy
        y = np.array(y)
        self.classes = np.unique(y)
        
        for c in self.classes:
            # Ambil subset data untuk setiap kelas
            X_c = X[y == c]
            
            # Hitung prior P(vj) untuk kelas c
            self.priors[c] = len(X_c) / len(y)
            
            # Hitung mean dan std dev untuk setiap fitur di kelas c
            self.mean_std[c]['mean'] = X_c.mean(axis=0)
            self.mean_std[c]['std'] = X_c.std(axis=0)
    
    def _calculate_likelihood(self, x, mean, std):
        # Menghitung distribusi Gaussian
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent
    
    def _calculate_posterior(self, x):
        posteriors = {}
        
        for c in self.classes:
            # Ambil prior P(vj)
            prior = self.priors[c]
            
            # Ambil mean dan std dev untuk kelas c
            mean = self.mean_std[c]['mean']
            std = self.mean_std[c]['std']
            
            # Hitung likelihood P(a1 | vj) * P(a2 | vj) * ... * P(an | vj)
            likelihood = np.prod(self._calculate_likelihood(x, mean, std))
            
            # Hitung posterior P(vj | a1, a2, ..., an)
            posteriors[c] = prior * likelihood
        
        return posteriors
    
    def predict(self, X):
        # Konversi X ke array numpy jika berbentuk DataFrame
        X = np.array(X)
        predictions = []
        for x in X:
            # Hitung posterior untuk setiap kelas
            posteriors = self._calculate_posterior(x)
            # Pilih kelas dengan posterior tertinggi
            predictions.append(max(posteriors, key=posteriors.get))
        return predictions
    
    def save_model(self, file_name):
        """Menyimpan model ke file."""
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved to {file_name}")
    
    @staticmethod
    def load_model(file_name):
        """Memuat model dari file."""
        with open(file_name, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {file_name}")
        return model
