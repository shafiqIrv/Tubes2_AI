import numpy as np
import pandas as pd
from collections import defaultdict
import pickle

class GaussianNaiveBayes:
    def __init__(self):
        # Inisialisasi variabel untuk menyimpan prior, mean, std, dan likelihood
        self.priors = {}
        self.mean_std = defaultdict(dict)
        self.classes = None
        self.precomputed_likelihoods = defaultdict(dict)
    
    def fit(self, X, y):
        # Pastikan y berbentuk array numpy
        y = np.array(y)
        self.classes = np.unique(y)
        self.features = X.columns
        
        # Hitung prior, mean, std, dan precompute likelihood untuk tiap kelas
        for c in self.classes:
            # Ambil data yang sesuai dengan kelas saat ini
            X_c = X[y == c]
            
            # Hitung prior P(vj) untuk kelas ini
            self.priors[c] = len(X_c) / len(y)
            
            # Hitung rata-rata (mean) dan deviasi standar (std) untuk tiap fitur
            self.mean_std[c]['mean'] = X_c.mean(axis=0).values
            self.mean_std[c]['std'] = X_c.std(axis=0).values
            
            # Precompute likelihood untuk setiap nilai unik di tiap fitur
            self.precomputed_likelihoods[c] = {}
            for i, feature in enumerate(self.features):
                # Ambil nilai-nilai unik dari fitur
                unique_values = np.unique(X[feature])
                
                # Hitung likelihood untuk setiap nilai unik
                mean = self.mean_std[c]['mean'][i]
                std = self.mean_std[c]['std'][i]
                self.precomputed_likelihoods[c][feature] = {
                    value: self._calculate_likelihood(value, mean, std)
                    for value in unique_values
                }
    
    def _calculate_likelihood(self, x, mean, std):
        # Hitung nilai likelihood menggunakan distribusi Gaussian
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent
    
    def predict(self, X):
        # Ubah X jadi array numpy kalau masih berbentuk DataFrame
        X = np.array(X)
        predictions = []
        
        # Loop untuk tiap data di X
        for x in X:
            posteriors = {}
            
            # Hitung posterior untuk tiap kelas
            for c in self.classes:
                # Mulai dengan prior P(vj)
                posterior = self.priors[c]
                
                # Kalikan dengan likelihood precomputed dari tiap fitur
                for i, feature in enumerate(self.features):
                    feature_value = x[i]
                    likelihood = self.precomputed_likelihoods[c][feature].get(feature_value, 1e-9)
                    posterior *= likelihood
                
                posteriors[c] = posterior
            
            # Tambahkan kelas dengan nilai posterior tertinggi
            predictions.append(max(posteriors, key=posteriors.get))
        
        return predictions
    
    def save_model(self, file_name):
        # Simpan model ke file dengan pickle
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model disimpan ke {file_name}")
    
    @staticmethod
    def load_model(file_name):
        # Load model dari file dengan pickle
        with open(file_name, 'rb') as file:
            model = pickle.load(file)
        print(f"Model dimuat dari {file_name}")
        return model
