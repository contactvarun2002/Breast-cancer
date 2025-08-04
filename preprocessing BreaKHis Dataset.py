import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from glob import glob

# ---------------------- Setup Paths ----------------------
input_folder = "/content/drive/MyDrive/Colab Notebooks/archive - 2025-08-01T210435.399/MIAS"  # This folder contains subfolders
output_folder = "/content/drive/MyDrive/Colab Notebooks/archive - 2025-08-01T210435.399/preprocessed_output"
os.makedirs(output_folder, exist_ok=True)

# ---------------------- Preprocessing Functions ----------------------
def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size)

def normalize_image(image):
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

def remove_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def remove_artifacts(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    cleaned = cv2.inpaint(image, thresh, 3, cv2.INPAINT_TELEA)
    return cleaned

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

# ---------------------- Preprocessing Pipeline ----------------------
def preprocess_image(image_path):
    original = cv2.imread(image_path)
    resized = resize_image(original)
    normalized = normalize_image(resized)
    denoised = remove_noise(normalized)
    artifact_removed = remove_artifacts(denoised)
    contrast_enhanced = enhance_contrast(artifact_removed)

    return {
        "Original": original,
        "Resized": resized,
        "Normalized": normalized,
        "Denoised": denoised,
        "Artifact Removed": artifact_removed,
        "Contrast Enhanced": contrast_enhanced
    }

# ---------------------- Run Batch Preprocessing ----------------------
# ✅ Collect all .png and .jpg images from all subfolders
image_paths = glob(os.path.join(input_folder, "**", "*.png"), recursive=True)
image_paths += glob(os.path.join(input_folder, "**", "*.jpg"), recursive=True)

for img_path in image_paths:
    # Get relative path to preserve subfolder structure
    relative_path = os.path.relpath(img_path, input_folder)
    output_path = os.path.join(output_folder, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = preprocess_image(img_path)
    final_output = results["Contrast Enhanced"]
    cv2.imwrite(output_path, final_output)

print(f"✅ {len(image_paths)} images preprocessed and saved to: {output_folder}")

# ---------------------- Show All Preprocessing Steps for Multiple Images ----------------------
def show_all_images(image_paths, max_images=10):
    for idx, img_path in enumerate(image_paths[:max_images]):
        print(f"\n🔍 Displaying image {idx+1}/{min(max_images, len(image_paths))}: {os.path.basename(img_path)}")
        results = preprocess_image(img_path)

        plt.figure(figsize=(18, 10))
        for i, (title, image) in enumerate(results.items()):
            plt.subplot(2, 3, i + 1)
            plt.title(title)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        plt.tight_layout()
        plt.show()

# ---------------------- Show First Few Images ----------------------
if image_paths:
    show_all_images(image_paths, max_images=10)  # You can change 10 to any other number