"""
utils/organize_dataset.py
Organize APTOS 2019 / EyePACS dataset into folder structure
required for Keras ImageDataGenerator.

Usage:
    python utils/organize_dataset.py \
        --csv   /path/to/train.csv \
        --img_dir /path/to/train_images/ \
        --out   dataset/ \
        --val_split 0.2

CSV format (APTOS 2019):
    id_code,diagnosis
    000c1434d8d7,0
    001639a390f0,2
    ...

Diagnosis → Folder mapping:
    0 → No_DR
    1 → Mild
    2 → Moderate
    3 → Severe
    4 → Proliferative_DR
"""

import os
import shutil
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

CLASS_MAP = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative_DR",
}


def organize(csv_path, img_dir, out_dir, val_split=0.2, seed=42):
    """Read CSV, split into train/val, copy images into class folders."""

    df = pd.read_csv(csv_path)
    print(f"[Dataset] Loaded CSV: {len(df)} rows")

    # Rename columns if needed (APTOS uses id_code / diagnosis)
    if "id_code" in df.columns:
        df = df.rename(columns={"id_code": "image", "diagnosis": "label"})
    elif "image" not in df.columns:
        raise ValueError("CSV must have 'id_code'/'diagnosis' or 'image'/'label' columns.")

    print(f"[Dataset] Class distribution:\n{df['label'].value_counts().to_string()}\n")

    # Train / Val split
    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=seed,
        stratify=df["label"]
    )
    print(f"[Dataset] Train: {len(train_df)}  Val: {len(val_df)}")

    # Copy images
    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        copied = 0
        skipped = 0
        for _, row in split_df.iterrows():
            label = int(row["label"])
            cls   = CLASS_MAP.get(label)
            if cls is None:
                continue

            # Try common image extensions
            fname = str(row["image"])
            src   = None
            for ext in ["", ".png", ".jpg", ".jpeg"]:
                candidate = os.path.join(img_dir, fname + ext)
                if os.path.exists(candidate):
                    src = candidate
                    break

            if src is None:
                skipped += 1
                continue

            dest_dir = os.path.join(out_dir, split_name, cls)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, os.path.basename(src))
            shutil.copy2(src, dest)
            copied += 1

        print(f"  [{split_name}] Copied: {copied}  Skipped (not found): {skipped}")

    print(f"\n[Dataset] ✓ Dataset organized at: {out_dir}")
    print("  Run: python train_model.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize DR dataset")
    parser.add_argument("--csv",       required=True, help="Path to train.csv")
    parser.add_argument("--img_dir",   required=True, help="Directory of images")
    parser.add_argument("--out",       default="dataset", help="Output directory")
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    organize(args.csv, args.img_dir, args.out, args.val_split)
