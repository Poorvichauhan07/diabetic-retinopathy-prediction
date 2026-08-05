"""
predict.py - Prediction Module
Loads the trained CNN model and runs inference on retinal images.
"""

import os
import numpy as np
import tensorflow as tf
from utils.image_processing import preprocess_for_prediction

# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join("trained_model", "dr_model.h5")

# Class names must match training order
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

# Human-readable labels
CLASS_LABELS = {
    "No_DR":            "No DR",
    "Mild":             "Mild",
    "Moderate":         "Moderate",
    "Severe":           "Severe",
    "Proliferative_DR": "Proliferative DR",
}

# Severity descriptions
SEVERITY_MAP = {
    "No_DR":            "No Diabetic Retinopathy",
    "Mild":             "Mild Non-Proliferative DR",
    "Moderate":         "Moderate Non-Proliferative DR",
    "Severe":           "Severe Non-Proliferative DR",
    "Proliferative_DR": "Proliferative DR (Vision Threatening)",
}

# Medical advice per class
ADVICE_MAP = {
    "No_DR": (
        "No diabetic retinopathy detected. Maintain good glycemic control. "
        "Schedule annual eye examinations. Keep HbA1c < 7% and blood pressure controlled."
    ),
    "Mild": (
        "Mild retinopathy detected. Improve blood sugar and blood pressure management. "
        "Schedule follow-up eye exam within 9 months. Review diabetes management plan."
    ),
    "Moderate": (
        "Moderate retinopathy detected. Consult ophthalmologist within 3–6 months. "
        "Strict glycemic control is critical. Consider referral for laser treatment evaluation."
    ),
    "Severe": (
        "Severe retinopathy detected. URGENT ophthalmology referral within 1 month. "
        "High risk of vision loss. Laser photocoagulation or anti-VEGF injections may be needed."
    ),
    "Proliferative_DR": (
        "CRITICAL: Proliferative DR detected. IMMEDIATE specialist referral required. "
        "Risk of blindness is high. Vitrectomy, laser surgery, or anti-VEGF therapy urgently needed."
    ),
}

# Risk level labels
RISK_MAP = {
    "No_DR":            ("Low",      "#2e7d32"),
    "Mild":             ("Low-Moderate", "#558b2f"),
    "Moderate":         ("Moderate", "#f57c00"),
    "Severe":           ("High",     "#d84315"),
    "Proliferative_DR": ("Critical", "#b71c1c"),
}


# ─── Model Loading ────────────────────────────────────────────────────────────

_model = None   # Cached model (loaded once)


def load_model():
    """Load the trained Keras model. Uses caching to avoid repeated loads."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Please run train_model.py first."
            )
        print(f"[Model] Loading from {MODEL_PATH} ...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("[Model] Loaded successfully.")
    return _model


# ─── Prediction ───────────────────────────────────────────────────────────────

def predict_image(image_path):
    """
    Run prediction on a single retinal image.

    Parameters:
        image_path (str): Full path to the uploaded image.

    Returns:
        dict: {
            "predicted_class":  str,   e.g. "Moderate"
            "class_key":        str,   e.g. "Moderate"
            "confidence":       float, e.g. 87.34
            "severity":         str,
            "advice":           str,
            "risk_level":       str,
            "risk_color":       str,
            "all_probs":        dict,  probabilities for all classes
        }
    """
    # Preprocess image
    img_array = preprocess_for_prediction(image_path)

    # Load model and run inference
    model = load_model()
    predictions = model.predict(img_array, verbose=0)[0]   # Shape: (5,)

    # Get top prediction
    class_index = int(np.argmax(predictions))
    class_key   = CLASS_NAMES[class_index]
    confidence  = float(predictions[class_index]) * 100.0

    # Build all probabilities dict
    all_probs = {
        CLASS_LABELS[cls]: round(float(predictions[i]) * 100, 2)
        for i, cls in enumerate(CLASS_NAMES)
    }

    risk_level, risk_color = RISK_MAP[class_key]

    return {
        "predicted_class":  CLASS_LABELS[class_key],
        "class_key":        class_key,
        "confidence":       round(confidence, 2),
        "severity":         SEVERITY_MAP[class_key],
        "advice":           ADVICE_MAP[class_key],
        "risk_level":       risk_level,
        "risk_color":       risk_color,
        "all_probs":        all_probs,
    }


def predict_batch(image_paths):
    """
    Run prediction on multiple images.

    Parameters:
        image_paths (list): List of image file paths.

    Returns:
        list of dicts (one per image).
    """
    results = []
    model   = load_model()

    for path in image_paths:
        try:
            result = predict_image(path)
            result["image_path"] = path
            results.append(result)
        except Exception as e:
            results.append({
                "image_path": path,
                "error": str(e)
            })

    return results


# ─── Demo / Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    print(f"\nRunning prediction on: {image_path}")
    result = predict_image(image_path)

    print("\n" + "=" * 50)
    print(f"  Predicted Class : {result['predicted_class']}")
    print(f"  Confidence      : {result['confidence']:.2f}%")
    print(f"  Severity        : {result['severity']}")
    print(f"  Risk Level      : {result['risk_level']}")
    print(f"\n  Advice:\n  {result['advice']}")
    print("\n  All Probabilities:")
    for cls, prob in result["all_probs"].items():
        bar = "█" * int(prob / 5)
        print(f"    {cls:<20} {prob:>6.2f}%  {bar}")
    print("=" * 50)
