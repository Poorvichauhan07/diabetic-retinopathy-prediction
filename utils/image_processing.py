"""
utils/image_processing.py
Image preprocessing utilities for the Diabetic Retinopathy System.
"""

import cv2
import numpy as np
from PIL import Image
import os

# Target image size expected by the CNN model
IMG_SIZE = (224, 224)


def load_image(filepath):
    """Load an image from disk using OpenCV."""
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Could not load image from: {filepath}")
    return img


def resize_image(img, size=IMG_SIZE):
    """Resize image to the target size."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def normalize_image(img):
    """
    Normalize pixel values to [0, 1].
    Also applies CLAHE for better retinal feature visibility.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_channel)

    # Merge back and convert to BGR
    enhanced = cv2.merge((l_clahe, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Normalize to [0, 1]
    normalized = enhanced_bgr.astype(np.float32) / 255.0
    return normalized


def preprocess_for_prediction(filepath):
    """
    Full preprocessing pipeline for a single image:
    Load -> Resize -> Normalize -> Add batch dimension.
    Returns a numpy array ready for model.predict().
    """
    img = load_image(filepath)
    img = resize_image(img, IMG_SIZE)
    img = normalize_image(img)
    img = np.expand_dims(img, axis=0)  # Shape: (1, 224, 224, 3)
    return img


def preprocess_dataset_image(filepath):
    """
    Preprocessing for training — returns (H, W, C) array.
    """
    img = load_image(filepath)
    img = resize_image(img, IMG_SIZE)
    img = normalize_image(img)
    return img


def apply_augmentation(img):
    """
    Apply random data augmentation to a training image.
    Augmentations: horizontal flip, vertical flip, rotation, brightness.
    """
    # Random horizontal flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)

    # Random vertical flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 0)

    # Random rotation (-20 to +20 degrees)
    angle = np.random.uniform(-20, 20)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))

    # Random brightness adjustment
    factor = np.random.uniform(0.8, 1.2)
    img = np.clip(img * factor, 0.0, 1.0)

    return img


def save_preprocessed_preview(filepath, output_path):
    """
    Save a preprocessed version of the image for preview purposes.
    """
    img = load_image(filepath)
    img = resize_image(img, (400, 400))
    cv2.imwrite(output_path, img)


def is_valid_image(filepath):
    """Check if the file is a valid image."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in valid_extensions:
        return False
    try:
        img = Image.open(filepath)
        img.verify()
        return True
    except Exception:
        return False
