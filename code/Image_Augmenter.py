import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Paths
IMAGE_DIR = "data"  # Directory containing images
LABEL_DIR = "label"  # Directory containing label files
FEATURES_OUTPUT = "features.npy"

# Supported image formats
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# Prepare the feature extraction model
def prepare_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}!")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
    return feature_extractor.to(device), device

# Preprocess image for the model
def preprocess_image(image_path, bbox=None):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')

    if bbox:
        # Crop the image using bounding box coordinates
        img_width, img_height = img.size
        x_center, y_center, width, height = bbox
        x_min = int((x_center - width / 2) * img_width)
        y_min = int((y_center - height / 2) * img_height)
        x_max = int((x_center + width / 2) * img_width)
        y_max = int((y_center + height / 2) * img_height)
        img = img.crop((x_min, y_min, x_max, y_max))

    return preprocess(img).unsqueeze(0)

# Parse label file
def parse_label_file(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    bboxes = []
    for line in lines:
        _, x_center, y_center, width, height = map(float, line.strip().split())
        bboxes.append((x_center, y_center, width, height))
    return bboxes

# Find the label file for a given image
def find_label_file(label_dir, base_name):
    label_path = os.path.join(label_dir, base_name + ".txt")
    return label_path if os.path.exists(label_path) else None

# Extract features
def extract_features(feature_extractor, image_dir, label_dir, output_file, device):
    features = {}
    image_files = [
        f for f in os.listdir(image_dir)
        if any(f.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)
    ]

    for image_file in tqdm(image_files, desc="Processing images", unit="image"):
        base_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(image_dir, image_file)
        label_path = find_label_file(label_dir, base_name)

        if not label_path:
            tqdm.write(f"Warning: No label file found for {image_file}")
            continue

        try:
            bboxes = parse_label_file(label_path)
            for i, bbox in enumerate(bboxes):
                input_tensor = preprocess_image(image_path, bbox).to(device)
                with torch.no_grad():
                    feature_vector = feature_extractor(input_tensor).flatten().cpu().numpy()
                features[f"{base_name}_bbox_{i}"] = feature_vector
        except Exception as e:
            tqdm.write(f"Error processing {image_file}: {e}")

    np.save(output_file, features)
    print(f"Features saved to {output_file}")

if __name__ == "__main__":
    # Prepare model
    feature_extractor, device = prepare_model()

    # Extract features
    extract_features(feature_extractor, IMAGE_DIR, LABEL_DIR, FEATURES_OUTPUT, device)
