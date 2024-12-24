import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.spatial import distance

categories = ['bird', 'car', 'cloud', 'dog', 'flower', 'house', 'human', 'mountain', 'sun', 'tree']

def load_data(features_file):
    data = np.load(features_file, allow_pickle=True).item()  # Load the dictionary
    X, y = [], []
    for key, feature in data.items():
        try:
            category = key.split('-')[1].split('_')[0]
            if category in categories:
                X.append(feature)
                y.append(categories.index(category))  # Convert category name to index
            else:
                print(f"Skipping key: {key} (Category not in predefined list)")
        except IndexError:
            print(f"Error parsing key: {key}")
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No data loaded. Please verify the format of `features.npy`.")
    return np.array(X), np.array(y)

def euclidean_distance(a, b):
    return distance.euclidean(a, b)

def manhattan_distance(a, b):
    return distance.cityblock(a, b)

def knn_predict(X_train, y_train, X_test, k=3, distance_metric='euclidean'):
    y_pred = []
    distance_function = euclidean_distance if distance_metric == 'euclidean' else manhattan_distance
    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train):
            dist = distance_function(test_point, train_point)
            distances.append((dist, y_train[i]))
        distances = sorted(distances, key=lambda x: x[0])
        k_nearest_labels = [label for _, label in distances[:k]]
        predicted_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        y_pred.append(predicted_label)
    return np.array(y_pred)

def main(features_file):
    X, y = load_data(features_file)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Predict using KNN
    k = 5  # Number of neighbors
    distance_metric = 'euclidean'  # Change to 'euclidean' or 'manhattan'
    y_pred = knn_predict(X_train, y_train, X_test, k=k, distance_metric=distance_metric)

    # Evaluate classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=categories))

if __name__ == "__main__":
    features_file = "features.npy"  # Path to features.npy
    main(features_file)
