import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image

# Configuration
categories = ['bird', 'car', 'cloud', 'dog', 'flower', 'house', 'human', 'mountain', 'sun', 'tree']
num_classes = len(categories)
img_size = 64  # Resize images to 64x64
batch_size = 32
epochs = 20

# Load and preprocess the dataset
def load_data(image_dir, label_dir):
    data = []
    labels = []
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        label_file = os.path.join(label_dir, img_file.replace('.png', '.txt'))
        try:
            # Parse label and bounding box from the corresponding label file
            with open(label_file, 'r') as f:
                label_data = f.readline().strip().split()
                label = int(label_data[0])
                x_center, y_center, width, height = map(float, label_data[1:])

            if 0 <= label < len(categories):
                # Load and crop the image using bounding box
                img = Image.open(img_path).convert('RGB')
                img_width, img_height = img.size

                x1 = int((x_center - width / 2) * img_width)
                x2 = int((x_center + width / 2) * img_width)
                y1 = int((y_center - height / 2) * img_height)
                y2 = int((y_center + height / 2) * img_height)

                img = img.crop((x1, y1, x2, y2))
                img = img.resize((img_size, img_size))

                data.append(np.array(img))
                labels.append(label)
        except Exception as e:
            print(f"Error processing {img_path} or {label_file}: {e}")
    return np.array(data), np.array(labels)

# Replace 'your_image_path' and 'your_label_path' with the actual paths to your dataset
image_dir = 'data'
label_dir = 'label'
data, labels = load_data(image_dir, label_dir)

# Normalize data and convert labels to one-hot encoding
data = data / 255.0
labels = to_categorical(labels, num_classes=num_classes)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Model definition
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model()
model.summary()

# Train the model
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    epochs=epochs
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Classification report
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred, target_names=categories))

# Save the model
model.save('hand_drawn_classifier.keras')



# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.show()
