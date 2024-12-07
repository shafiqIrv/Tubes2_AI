# Import from classes.py
from classes import *

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss


# Dataset tetap
X = np.array([
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0],
    [0, 1, 0, 0, 1]
])
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# Split dataset menjadi train dan test
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Latih model
model = ID3Classifier()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Konversi probabilitas ke format yang kompatibel
y_proba_formatted = y_proba

# Evaluasi
accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_proba_formatted)

print(f"Accuracy: {accuracy}")
print(f"Log Loss: {logloss}")
# Output:
print(f"Prediksi: {y_pred}")
print(f"Probabilitas: {y_proba}")
