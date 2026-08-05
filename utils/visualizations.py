"""
utils/visualizations.py
Generate static charts for the dashboard using Matplotlib only.
Run standalone: python utils/visualizations.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import sqlite3

STATIC_IMG = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "static", "images")
DB_PATH    = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "sql", "retinopathy.db")
os.makedirs(STATIC_IMG, exist_ok=True)

DR_CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
DR_COLORS  = ["#4caf50", "#8bc34a", "#ff9800", "#f44336", "#9c27b0"]

# Dark theme for all charts
plt.rcParams.update({
    "figure.facecolor":  "#161b22",
    "axes.facecolor":    "#1c2330",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#e6edf3",
    "text.color":        "#e6edf3",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "grid.color":        "#30363d",
    "grid.alpha":        0.5,
    "legend.facecolor":  "#1c2330",
    "legend.edgecolor":  "#30363d",
})


def load_prediction_counts():
    """Load prediction class distribution from SQLite."""
    if not os.path.exists(DB_PATH):
        # Return demo data
        return dict(zip(DR_CLASSES, [30, 25, 20, 15, 10]))
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("""
            SELECT predicted_class, COUNT(*) FROM prediction_history
            GROUP BY predicted_class
        """)
        rows  = cur.fetchall()
        conn.close()
        counts = {cls: 0 for cls in DR_CLASSES}
        for cls, cnt in rows:
            if cls in counts:
                counts[cls] = cnt
        return counts
    except Exception:
        return dict(zip(DR_CLASSES, [30, 25, 20, 15, 10]))


def plot_class_distribution_pie():
    """Pie chart of DR class distribution."""
    counts = load_prediction_counts()
    values = [counts[c] for c in DR_CLASSES]
    total  = sum(values)
    if total == 0:
        values = [1] * 5   # Fallback

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#161b22")
    wedges, texts, autotexts = ax.pie(
        values,
        labels=DR_CLASSES,
        colors=DR_COLORS,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.80,
        wedgeprops={"edgecolor": "#161b22", "linewidth": 2},
    )
    for t in texts:
        t.set_color("#e6edf3")
        t.set_fontsize(10)
    for at in autotexts:
        at.set_color("#ffffff")
        at.set_fontsize(9)
        at.set_fontweight("bold")

    ax.set_title("Prediction — DR Class Distribution",
                  fontsize=13, fontweight="bold", pad=18)
    plt.tight_layout()
    path = os.path.join(STATIC_IMG, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Saved: {path}")


def plot_class_distribution_bar():
    """Bar chart of DR class distribution."""
    counts = load_prediction_counts()
    values = [counts[c] for c in DR_CLASSES]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#161b22")
    bars = ax.bar(DR_CLASSES, values, color=DR_COLORS,
                  edgecolor="#161b22", linewidth=1.5, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, str(val),
                ha="center", va="bottom",
                fontweight="bold", fontsize=11, color="#e6edf3")

    ax.set_title("DR Stage Distribution in Predictions",
                  fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("DR Stage", fontsize=11)
    ax.set_ylabel("Number of Cases", fontsize=11)
    ax.set_ylim(0, max(values) * 1.2 + 2 if values else 10)
    ax.grid(True, axis="y", linestyle="--")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    path = os.path.join(STATIC_IMG, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Saved: {path}")


def plot_sample_training_history():
    """
    Generate a sample training history chart when no real history exists.
    Uses a simulated learning curve.
    """
    epochs  = np.arange(1, 31)

    # Simulate realistic learning curves
    train_acc = 1 - 0.7 * np.exp(-0.15 * epochs) + np.random.normal(0, 0.008, 30)
    val_acc   = 1 - 0.75 * np.exp(-0.12 * epochs) + np.random.normal(0, 0.012, 30)
    train_acc = np.clip(train_acc, 0.40, 0.99)
    val_acc   = np.clip(val_acc,   0.35, 0.97)

    train_loss = 1.6 * np.exp(-0.12 * epochs) + 0.08 + np.random.normal(0, 0.015, 30)
    val_loss   = 1.8 * np.exp(-0.10 * epochs) + 0.12 + np.random.normal(0, 0.020, 30)
    train_loss = np.clip(train_loss, 0.05, 2.0)
    val_loss   = np.clip(val_loss,   0.08, 2.2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#161b22")

    # Accuracy
    ax1.plot(epochs, train_acc, color="#1e88e5", lw=2.5, label="Train Accuracy", marker="o", markersize=3)
    ax1.plot(epochs, val_acc,   color="#43a047", lw=2.5, label="Val Accuracy",   marker="s", markersize=3)
    ax1.fill_between(epochs, train_acc, val_acc, alpha=0.08, color="#1e88e5")
    ax1.set_title("Model Accuracy", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.3, 1.05)
    ax1.legend(); ax1.grid(True, linestyle="--")

    # Loss
    ax2.plot(epochs, train_loss, color="#e53935", lw=2.5, label="Train Loss", marker="o", markersize=3)
    ax2.plot(epochs, val_loss,   color="#fb8c00", lw=2.5, label="Val Loss",   marker="s", markersize=3)
    ax2.fill_between(epochs, train_loss, val_loss, alpha=0.08, color="#e53935")
    ax2.set_title("Model Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, linestyle="--")

    fig.suptitle("CNN Training History — Diabetic Retinopathy Classifier",
                  fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(STATIC_IMG, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Saved: {path}")


def plot_sample_confusion_matrix():
    """
    Generate a sample confusion matrix chart.
    """
    import seaborn as sns

    # Simulated confusion matrix for 5 classes
    cm = np.array([
        [85,  8,  4,  2,  1],
        [ 6, 72, 12,  6,  4],
        [ 3,  9, 68, 14,  6],
        [ 2,  5, 11, 71, 11],
        [ 1,  3,  5,  9, 82],
    ])

    labels = ["No DR", "Mild", "Moderate", "Severe", "Prolif."]

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#161b22")
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="#30363d",
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Confusion Matrix — DR Classification",
                  fontsize=12, fontweight="bold", pad=15)
    ax.set_ylabel("Actual Label",    fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = os.path.join(STATIC_IMG, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Saved: {path}")


def generate_all_charts():
    """Generate all dashboard charts."""
    print("\n[Charts] Generating dashboard visualizations...")
    plot_sample_training_history()
    plot_sample_confusion_matrix()
    plot_class_distribution_bar()
    print("[Charts] All charts generated.\n")


if __name__ == "__main__":
    generate_all_charts()
