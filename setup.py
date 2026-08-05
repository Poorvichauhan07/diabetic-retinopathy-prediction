"""
setup.py — One-Click Project Setup
Initializes database, generates demo charts, and seeds demo data.
Run ONCE after installing requirements:
    python setup.py
"""

import os
import sys

def run(cmd):
    print(f"\n$ {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"[WARNING] Command exited with code {ret}")
    return ret


def main():
    print("\n" + "=" * 60)
    print("  Diabetic Retinopathy System — Project Setup")
    print("=" * 60)

    # 1. Create all required directories
    dirs = [
        "uploads", "reports", "trained_model",
        "dataset/train", "dataset/val",
        "sql", "static/images", "static/css", "static/js",
        "templates", "utils",
    ]
    print("\n[1/4] Creating directory structure...")
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  ✓  {d}/")

    # 2. Initialize database
    print("\n[2/4] Initializing SQLite database...")
    import database as db
    print("  ✓  Database initialized at sql/retinopathy.db")

    # 3. Seed demo data
    print("\n[3/4] Seeding demo data...")
    try:
        from utils.demo_data import seed
        seed()
    except Exception as e:
        print(f"  [WARN] Demo data seeding failed: {e}")

    # 4. Generate placeholder charts
    print("\n[4/4] Generating dashboard charts...")
    try:
        from utils.visualizations import generate_all_charts
        generate_all_charts()
    except ImportError as e:
        print(f"  [WARN] Chart generation skipped: {e}")
        print("  Install seaborn: pip install seaborn --break-system-packages")

    print("\n" + "=" * 60)
    print("  ✅  Setup complete!")
    print()
    print("  Next steps:")
    print("  1. Run the app:    python main.py")
    print("  2. Open browser:   http://localhost:5000")
    print()
    print("  To train the model (optional):")
    print("  3. Add dataset to dataset/train/ and dataset/val/")
    print("     (See README.md → Dataset Setup)")
    print("  4. Train:          python train_model.py")
    print()
    print("  To predict from CLI:")
    print("  5. python predict.py uploads/your_image.jpg")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
