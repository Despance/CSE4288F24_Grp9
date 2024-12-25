import tensorflow as tf
import cv2
import numpy as np
import os

# Constants
MODEL_PATH = 'child_drawing_classifier_with_aug.keras'
IMAGE_FOLDER = 'test_images'  # Change this to your image folder path
IMG_SIZE = 256

# Define Categories
CATEGORIES = ['bird', 'car', 'cloud', 'dog', 'flower', 'house', 'human', 'mountain', 'sun', 'tree']

# Load the Saved Model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
    return model

# Preprocess the Image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"❌ Image not found at path: {image_path}")
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Predict the Image Category
def predict_image(model, image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    predicted_category = CATEGORIES[class_index]
    
    print(f"🔍 Prediction: {os.path.basename(image_path)} → {predicted_category} ({confidence:.2f})")
    return predicted_category, confidence

# Predict All Images in Folder
def predict_images_in_folder(model, folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"❌ Folder not found at path: {folder_path}")
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
    if not image_files:
        print("⚠️ No image files found in the folder.")
        return
    
    print(f"📂 Found {len(image_files)} image(s) in '{folder_path}'")
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        try:
            predict_image(model, image_path)
        except Exception as e:
            print(f"❌ Error processing {image_file}: {e}")

# Main Execution
if __name__ == '__main__':
    model = load_model(MODEL_PATH)
    predict_images_in_folder(model, IMAGE_FOLDER)
