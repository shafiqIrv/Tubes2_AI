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
        self.classes = np.unique(y)
        
        for c in self.classes:
            X_c = X[y == c]
            
            # Menghitung P(vj)
            self.priors[c] = len(X_c) / len(y)
            
            # Menghitung mean dan std dev setiap fitur
            self.mean_std[c]['mean'] = np.mean(X_c, axis=0)
            self.mean_std[c]['std'] = np.std(X_c, axis=0)
    
    def _calculate_likelihood(self, x, mean, std):
        # Menghitung distribusi Gaussian
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent
    
    def _calculate_posterior(self, x):
        posteriors = {}
        
        for c in self.classes:
            # Menghitung P(vj)
            prior = self.priors[c]
            
            # Menghitung P(a1 | vj) * P(a2 | vj) * ... * P(an | vj)
            mean = self.mean_std[c]['mean']
            std = self.mean_std[c]['std']
            likelihood = np.prod(self._calculate_likelihood(x, mean, std))
            
            # Menghitung posterior P(vj | a1, a2, ..., an) = P(vj) * P(a1 | vj) * ... * P(an | vj)
            posteriors[c] = prior * likelihood
        
        return posteriors
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = self._calculate_posterior(x)
            # Pilih kelas yang posterior tertinggi
            predictions.append(max(posteriors, key=posteriors.get))
        return predictions
    
    def save_model(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved to {file_name}")
    
    @staticmethod
    def load_model(file_name):
        with open(file_name, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {file_name}")
        return model
