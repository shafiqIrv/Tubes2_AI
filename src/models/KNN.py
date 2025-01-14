import numpy as np
from collections import Counter
import pickle

def distance(A, B, p):
    # A dan B array point, p parameter (lihat __init__)
    return np.sum(np.abs(A - B) ** p) ** (1 / p)

class KNN_FromScratch:
    def __init__(self, n_neighbors: int, p: float):
        # p: parameter angka untuk menghitung distance Minkowski
        # 1 untuk manhattan, 2 untuk euclidean, minkowski terserah mau masukin angka berapa (jangan yg aneh aneh)
        self.n_neighbors = n_neighbors
        self.p = p

    def fit(self, X_train, Y_train):
        # asumsi scaling dan encoding sudah di notebook
        # fungsi ini cuma simpan training data
        self.X_train = X_train.to_numpy()
        self.Y_train = Y_train.to_numpy()
        return
    
    def predict_instance(self, instance):
        distances = [distance(i, instance.to_numpy(), self.p) for i in self.X_train]
        nearest_neighbors_index = np.argsort(distances)[:self.n_neighbors] # argsort yg direturn indexnya
        nearest_neighbors  = self.Y_train[nearest_neighbors_index]
        nearest_neighbors = [label.item() if isinstance(label, np.ndarray) else label for label in nearest_neighbors]
        return Counter(nearest_neighbors).most_common(1)[0][0] # cari yg paling banyak
    
    def predict(self, X_test):
        # X_test itu semua data test
        return [self.predict_instance(instance) for _, instance in X_test.iterrows()]

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

# test
# if __name__ == "__main__":
#     X = np.array([
#         [0, 1, 0, 1, 0],
#         [1, 0, 1, 0, 1],
#         [1, 1, 0, 1, 0],
#         [0, 0, 1, 0, 1],
#         [1, 0, 0, 1, 0],
#         [0, 1, 1, 0, 1],
#         [1, 1, 1, 1, 0],
#         [0, 0, 0, 0, 1],
#         [1, 0, 1, 1, 0],
#         [0, 1, 0, 0, 1]
#     ])
#     y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

#     # Split dataset menjadi train dan test
#     train_size = int(0.8 * len(X))
#     X_train, X_test = X[:train_size], X[train_size:]
#     y_train, y_test = y[:train_size], y[train_size:]

#     # Latih model
#     model = KNN_FromScratch(2, 1)
#     model.fit(X_train, y_train)

#     # Prediksi
#     y_pred = model.predict(X_test)
#     print(y_pred)