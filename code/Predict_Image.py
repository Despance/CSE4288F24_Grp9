import os
import numpy as np
from keras.models import load_model
from PIL import Image

# --- Configuration ---
MODEL_PATH = 'hand_drawn_classifier.keras'
IMG_SIZE = 256  # Image size used during training
TEST_IMAGES_DIR = 'test_images'  # Folder containing test images
CATEGORIES = ['bird', 'car', 'cloud', 'dog', 'flower', 'house', 'human', 'mountain', 'sun', 'tree']

# --- Load the Model ---
def load_cnn_model(model_path):
    """Load the trained Keras CNN model."""
    return load_model(model_path)

# --- Preprocess an Image ---
def preprocess_image(image_path, img_size=IMG_SIZE):
    """
    Load, resize, and normalize an image.
    Args:
        image_path (str): Path to the image file.
        img_size (int): Target image size.
    Returns:
        np.array: Preprocessed image ready for prediction.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# --- Predict Image Class ---
def predict_image(model, image_path):
    """
    Predict the class of an image using the trained model.
    Args:
        model: Trained Keras model.
        image_path (str): Path to the image file.
    Returns:
        str: Predicted class label.
        float: Confidence score.
    """
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]
    return CATEGORIES[predicted_class], confidence

# --- Process All Images in a Folder ---
def classify_images_in_folder(model, folder_path):
    """
    Classify all images in a specified folder.
    Args:
        model: Trained Keras model.
        folder_path (str): Path to the folder containing images.
    """
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    if not image_files:
        print("No image files found in the folder.")
        return

    print(f"Found {len(image_files)} image(s) in '{folder_path}'")
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        try:
            label, confidence = predict_image(model, image_path)
            print(f"{image_file}: Predicted Class: {label}, Confidence: {confidence:.2f}")
        except Exception as e:
            print(f"Failed to process {image_file}: {e}")

# --- Main Function ---
if __name__ == '__main__':
    model = load_cnn_model(MODEL_PATH)
    classify_images_in_folder(model, TEST_IMAGES_DIR)
