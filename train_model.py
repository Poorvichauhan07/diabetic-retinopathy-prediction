"""
train_model.py - CNN Model Training Script
Trains a Convolutional Neural Network for Diabetic Retinopathy classification.

Dataset expected structure:
    dataset/
        train/
            No_DR/      (class 0)
            Mild/       (class 1)
            Moderate/   (class 2)
            Severe/     (class 3)
            Proliferative_DR/  (class 4)
        val/
            (same structure)

Run: python train_model.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (for saving plots)
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout,
                                      Flatten, BatchNormalization,
                                      GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                         ReduceLROnPlateau)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

# ─── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-4
NUM_CLASSES = 5

DATASET_DIR   = os.path.join("dataset")
TRAIN_DIR     = os.path.join(DATASET_DIR, "train")
VAL_DIR       = os.path.join(DATASET_DIR, "val")
MODEL_DIR     = "trained_model"
MODEL_PATH    = os.path.join(MODEL_DIR, "dr_model.h5")
STATIC_DIR    = os.path.join("static", "images")

CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]
CLASS_LABELS = {
    "No_DR": "No DR",
    "Mild": "Mild",
    "Moderate": "Moderate",
    "Severe": "Severe",
    "Proliferative_DR": "Proliferative DR"
}

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


# ─── Data Generators ──────────────────────────────────────────────────────────

def create_data_generators():
    """
    Create training and validation data generators with augmentation.
    """
    # Training augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        shear_range=0.1,
        fill_mode="nearest"
    )

    # Validation — only rescale, no augmentation
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False
    )

    return train_gen, val_gen


# ─── Model Architecture ───────────────────────────────────────────────────────

def build_model():
    """
    Build CNN model using MobileNetV2 as backbone (Transfer Learning).
    MobileNetV2 is lightweight and effective for medical image classification.
    """
    # Load pre-trained MobileNetV2 without top layers
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze base model layers initially
    base_model.trainable = False

    # Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(model.summary())
    return model, base_model


def unfreeze_and_finetune(model, base_model):
    """
    Unfreeze the top layers of the base model for fine-tuning.
    """
    # Unfreeze the last 30 layers
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=LR / 10),  # Lower LR for fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    """Main training function."""
    print("\n" + "=" * 60)
    print("  Diabetic Retinopathy CNN Model Training")
    print("=" * 60)

    # Check dataset
    if not os.path.exists(TRAIN_DIR):
        print(f"\n[ERROR] Training directory not found: {TRAIN_DIR}")
        print("Please set up the dataset directory structure as described in README.md")
        print("\nCreating demo dataset for testing...")
        create_demo_dataset()

    # Create generators
    print("\n[1/5] Loading data generators...")
    train_gen, val_gen = create_data_generators()
    print(f"      Training samples : {train_gen.n}")
    print(f"      Validation samples: {val_gen.n}")
    print(f"      Class mapping     : {train_gen.class_indices}")

    # Build model
    print("\n[2/5] Building CNN model (MobileNetV2 backbone)...")
    model, base_model = build_model()

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Phase 1: Train with frozen base
    print("\n[3/5] Phase 1 — Training classification head (base model frozen)...")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS // 2,
        callbacks=callbacks,
        verbose=1
    )

    # Phase 2: Fine-tune with unfrozen layers
    print("\n[4/5] Phase 2 — Fine-tuning (top layers unfrozen)...")
    model = unfreeze_and_finetune(model, base_model)
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        initial_epoch=len(history1.history["loss"]),
        callbacks=callbacks,
        verbose=1
    )

    # Merge histories
    history = {}
    for key in history1.history:
        history[key] = history1.history[key] + history2.history.get(key, [])

    # Evaluation
    print("\n[5/5] Evaluating model...")
    evaluate_model(model, val_gen, history)

    print(f"\n[✓] Model saved to: {MODEL_PATH}")
    return model


def evaluate_model(model, val_gen, history):
    """Run evaluation, generate plots and classification report."""
    val_gen.reset()

    # Predictions
    y_pred_probs = model.predict(val_gen, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = val_gen.classes

    # Classification report
    print("\n--- Classification Report ---")
    report = classification_report(y_true, y_pred,
                                   target_names=list(CLASS_LABELS.values()))
    print(report)

    # Save report to file
    with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    # Accuracy & Loss plots
    plot_training_history(history)

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred)

    # Disease distribution
    plot_class_distribution(val_gen)


def plot_training_history(history):
    """Plot and save accuracy and loss graphs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Training History", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history["accuracy"],     label="Train Accuracy",  color="#1e88e5", lw=2)
    axes[0].plot(history["val_accuracy"], label="Val Accuracy",    color="#43a047", lw=2)
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history["loss"],     label="Train Loss",  color="#e53935", lw=2)
    axes[1].plot(history["val_loss"], label="Val Loss",    color="#fb8c00", lw=2)
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(STATIC_DIR, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Training history plot saved: {path}")


def plot_confusion_matrix(y_true, y_pred):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    labels = list(CLASS_LABELS.values())

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="gray")
    plt.title("Confusion Matrix — Diabetic Retinopathy Classification",
              fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Actual Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = os.path.join(STATIC_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Confusion matrix saved: {path}")


def plot_class_distribution(val_gen):
    """Plot class distribution in the validation set."""
    labels = list(CLASS_LABELS.values())
    counts = np.bincount(val_gen.classes, minlength=len(labels))

    colors_list = ["#43a047", "#1e88e5", "#fb8c00", "#e53935", "#8e24aa"]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, counts, color=colors_list, edgecolor="white", linewidth=1.5)

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5, str(count),
                 ha="center", va="bottom", fontweight="bold")

    plt.title("Validation Set — Disease Class Distribution",
              fontsize=13, fontweight="bold")
    plt.xlabel("DR Stage", fontsize=11)
    plt.ylabel("Number of Images", fontsize=11)
    plt.xticks(rotation=15, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(STATIC_DIR, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Class distribution plot saved: {path}")


# ─── Demo Dataset Creator ─────────────────────────────────────────────────────

def create_demo_dataset():
    """
    Create a small synthetic dataset for testing without downloading APTOS.
    Generates random coloured images per class (50 train + 10 val each).
    """
    import cv2

    class_colors = {
        "No_DR":            (180, 220, 180),
        "Mild":             (180, 200, 220),
        "Moderate":         (220, 200, 160),
        "Severe":           (220, 170, 150),
        "Proliferative_DR": (200, 150, 150),
    }

    for split, n in [("train", 50), ("val", 10)]:
        for cls, color in class_colors.items():
            cls_dir = os.path.join(DATASET_DIR, split, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(n):
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                # Simulate retinal fundus image with circular pattern
                cv2.circle(img, (112, 112), 100, color, -1)
                cv2.circle(img, (112, 112), 30, (255, 200, 100), -1)
                # Add random noise
                noise = np.random.randint(0, 40, img.shape, dtype=np.uint8)
                img = cv2.add(img, noise)
                cv2.imwrite(os.path.join(cls_dir, f"{cls}_{i:03d}.png"), img)

    print("[✓] Demo dataset created in dataset/")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = train()
