import numpy as np
import pandas as pd

class ID3Classifier:
    def __init__(self):
        self.tree = None

    def _entropy(self, y):
        """Menghitung entropy dari array label."""
        values, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _information_gain(self, X, y, feature):
        """Menghitung information gain dari fitur tertentu."""
        total_entropy = self._entropy(y)
        values, counts = np.unique(X[:, feature], return_counts=True)
        weighted_entropy = np.sum(
            (counts[i] / len(y)) * self._entropy(y[X[:, feature] == values[i]])
            for i in range(len(values))
        )
        return total_entropy - weighted_entropy

    def _best_feature(self, X, y):
        """Menentukan fitur terbaik untuk split berdasarkan information gain."""
        return np.argmax([self._information_gain(X, y, i) for i in range(X.shape[1])])

    def _build_tree(self, X, y):
        """Rekursif membangun pohon keputusan."""
        # Jika semua label sama, kembalikan distribusi label
        if len(np.unique(y)) == 1:
            return {"label": y[0], "proba": {y[0]: 1.0}}

        # Jika tidak ada fitur untuk split, kembalikan distribusi label
        if X.shape[1] == 0:
            counts = np.bincount(y)
            probs = counts / len(y)
            return {"label": counts.argmax(), "proba": dict(enumerate(probs))}

        # Tentukan fitur terbaik untuk split
        best_feat = self._best_feature(X, y)
        tree = {best_feat: {}}

        # Split data berdasarkan nilai fitur terbaik
        for value in np.unique(X[:, best_feat]):
            sub_X = X[X[:, best_feat] == value]
            sub_y = y[X[:, best_feat] == value]
            subtree = self._build_tree(np.delete(sub_X, best_feat, axis=1), sub_y)
            tree[best_feat][value] = subtree

        return tree

    def fit(self, X, y):
        """Melatih model ID3 berdasarkan fitur dan label."""
        self.tree = self._build_tree(np.array(X), np.array(y))

    def _predict_one(self, sample, tree):
        """Memprediksi label untuk satu sample."""
        if not isinstance(tree, dict) or "proba" in tree:
            return tree["label"]

        feature = next(iter(tree))
        value = sample[feature]
        subtree = tree[feature].get(value, None)
        if subtree is None:
            return None  # Jika tidak ada nilai yang sesuai

        return self._predict_one(sample, subtree)

    def predict(self, X):
        """Memprediksi label untuk data baru."""
        return np.array([self._predict_one(sample, self.tree) for sample in np.array(X)])

    def _predict_proba_one(self, sample, tree):
        """Menghitung probabilitas untuk satu sample."""
        if not isinstance(tree, dict) or "proba" in tree:
            return tree["proba"]

        feature = next(iter(tree))
        value = sample[feature]
        subtree = tree[feature].get(value, None)
        if subtree is None:
            return None  # Jika tidak ada nilai yang sesuai

        return self._predict_proba_one(sample, subtree)

    def predict_proba(self, X):
        """Menghitung probabilitas untuk data baru."""
        probas = []
        for sample in np.array(X):
            proba = self._predict_proba_one(sample, self.tree)
            if proba is None:
                proba = {k: 0 for k in range(len(np.unique(y)))}  # Default proba jika tidak ditemukan
            probas.append(proba)
            
        return [list(proba.values()) for proba in probas]
