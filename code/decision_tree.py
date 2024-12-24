import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

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

def train_decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def main(features_file):
    X, y = load_data(features_file)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train Decision Tree
    classifier = train_decision_tree(X_train, y_train)

    # Test classifier
    y_pred = classifier.predict(X_test)

    # Evaluate classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=categories))

if __name__ == "__main__":
    features_file = "features.npy"  # Path to features.npy
    main(features_file)
