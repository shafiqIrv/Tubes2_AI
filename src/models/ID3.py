import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Fungsi untuk menghitung entropy
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

# Fungsi untuk menghitung information gain
def information_gain(data, feature, target):
    total_entropy = entropy(target)
    
    # Nilai threshold terbaik berdasarkan informasi maksimum
    best_threshold = None
    best_gain = -1

    unique_values = np.unique(data[feature])
    for threshold in unique_values:
        left_split = target[data[feature] <= threshold]
        right_split = target[data[feature] > threshold]
        
        # Hitung entropy weighted split
        weight_left = len(left_split) / len(target)
        weight_right = len(right_split) / len(target)
        
        split_entropy = (weight_left * entropy(left_split)) + (weight_right * entropy(right_split))
        
        # Gain = Entropy awal - Entropy split
        gain = total_entropy - split_entropy

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_gain, best_threshold

# Fungsi untuk membangun pohon decision tree
class ID3Numeric:
    def __init__(self, max_depth=None):
        self.tree = None
        self.max_depth = max_depth

    def fit(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return np.unique(y)[0]  # Return leaf node dengan kelas dominan

        best_feature = None
        best_threshold = None
        best_gain = -1

        for feature in X.columns:
            gain, threshold = information_gain(X, feature, y)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

        if best_gain == 0:
            return np.unique(y)[0]

        tree = {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': None,
            'right': None
        }

        left_indices = X[best_feature] <= best_threshold
        right_indices = X[best_feature] > best_threshold

        tree['left'] = self.fit(X[left_indices], y[left_indices], depth + 1)
        tree['right'] = self.fit(X[right_indices], y[right_indices], depth + 1)

        return tree

    def predict_instance(self, instance, tree):
        if not isinstance(tree, dict):
            return tree

        feature = tree['feature']
        threshold = tree['threshold']

        if instance[feature] <= threshold:
            return self.predict_instance(instance, tree['left'])
        else:
            return self.predict_instance(instance, tree['right'])

    def predict(self, X):
        return [self.predict_instance(row, self.tree) for _, row in X.iterrows()]


if __name__ == "__main__":
    # Dataset manual
    data = {
        'Feature1': [2, 3, 1, 5, 7, 9, 6, 8, 3, 2],
        'Feature2': [1, 3, 2, 4, 5, 7, 6, 9, 3, 1],
        'Feature3': [5, 7, 8, 2, 3, 4, 6, 7, 5, 8],
        'Target': [0, 1, 0, 1, 1, 0, 1, 1, 0, 0]
    }

    df_additional_features_train = pd.read_csv('data/train/additional_features_train.csv')
    df_labels_train = pd.read_csv('data/train/labels_train.csv')
    df_basic_features_train = pd.read_csv('data/train/basic_features_train.csv')
    df_content_features_train = pd.read_csv('data/train/content_features_train.csv')
    df_flow_features_train = pd.read_csv('data/train/flow_features_train.csv')
    df_time_features_train = pd.read_csv('data/train/time_features_train.csv')


    df_train = pd.merge(df_additional_features_train, df_basic_features_train, on='id')
    df_train = pd.merge(df_train, df_basic_features_train, on='id')
    df_train = pd.merge(df_train, df_labels_train, on='id')
    df_train = pd.merge(df_train, df_content_features_train, on='id')
    df_train = pd.merge(df_train, df_flow_features_train, on='id')
    df_train = pd.merge(df_train, df_time_features_train, on='id')



    df= df_train.head(10000)
    df = df.dropna()
    # Split dataset menjadi train dan validate

    # Fitur dan Target
    print(df.columns)

    # filter x where X is non object
    X = df.select_dtypes(exclude=['object'])
    X = df.drop(columns=['id'])
    # X = X.dropna()
    y = df['attack_cat']


    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

    print("Fitur X:\n", X.head())
    print("Target y:\n", y.head())


    # # Inisialisasi dan training ID3w
    id3 = ID3Numeric(max_depth=3)
    id3.tree = id3.fit(X_train, y_train)

    # # Cetak tree hasil training
    # print("Decision Tree:")
    # print(id3.tree)

    # # Prediksi
    predictions = id3.predict(X_test)

    # Print accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

    # print("Predictions:", predictions)
