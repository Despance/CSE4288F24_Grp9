import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Constants
IMAGE_DIR = 'data'
LABEL_DIR = 'label'
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 30

# Define Categories
CATEGORIES = ['bird', 'car', 'cloud', 'dog', 'flower', 'house', 'human', 'mountain', 'sun', 'tree']
NUM_CLASSES = len(CATEGORIES)

# Load Data with Cropping
def load_data(image_dir, label_dir):
    images = []
    labels = []

    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.jpg', '.png')):
            base_name = os.path.splitext(img_file)[0]
            label_file = os.path.join(label_dir, base_name + '.txt')
            
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    label_data = f.readline().split()
                    class_id = int(label_data[0])
                    x, y, w, h = map(float, label_data[1:])
                    
                    # Load and preprocess image
                    img_path = os.path.join(image_dir, img_file)
                    img = cv2.imread(img_path)
                    h_img, w_img, _ = img.shape
                    x1 = int((x - w / 2) * w_img)
                    y1 = int((y - h / 2) * h_img)
                    x2 = int((x + w / 2) * w_img)
                    y2 = int((y + h / 2) * h_img)
                    
                    cropped_img = img[y1:y2, x1:x2]
                    resized_img = cv2.resize(cropped_img, (IMG_SIZE, IMG_SIZE))
                    normalized_img = resized_img / 255.0

                    images.append(normalized_img)
                    labels.append(class_id)
    
    return np.array(images), np.array(labels)

# Load dataset
images, labels = load_data(IMAGE_DIR, LABEL_DIR)
labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data Augmentation
data_gen = ImageDataGenerator(
    rotation_range=20,       # Rotate images by up to 20 degrees
    width_shift_range=0.2,   # Shift width by up to 20%
    height_shift_range=0.2,  # Shift height by up to 20%
    shear_range=0.2,         # Shear by up to 20%
    zoom_range=0.2,          # Zoom by up to 20%
    horizontal_flip=True,    # Flip images horizontally
    fill_mode='nearest'      # Fill missing pixels with nearest values
)

# Generate augmented training data
train_data_gen = data_gen.flow(X_train, y_train, batch_size=BATCH_SIZE)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model with Augmented Data
history = model.fit(train_data_gen, 
                    epochs=EPOCHS, 
                    validation_data=(X_test, y_test))

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Predictions Example
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    return CATEGORIES[class_index]

# Example usage
# print(predict_image('path_to_sample_image.jpg'))


# Generate Predictions on Test Set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Print Classification Report
print("Classification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=CATEGORIES))


# Plot Training History
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Save Model
model.save('child_drawing_classifier_with_aug.keras')
