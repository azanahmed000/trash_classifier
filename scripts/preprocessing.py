"""
preprocessing.py
=================
Data Preprocessing Script for Trash Classifier

This script handles the splitting of the raw image dataset into
training and validation sets. It reads images from a source directory
organized by class (e.g., battery/, biological/, plastic/) and copies
them into separate train/ and val/ directories using an 80/20 split ratio.

Usage:
    python preprocessing.py

Directory Structure (Before):
    data set/
    ├── battery/
    ├── biological/
    ├── brown-glass/
    ├── cardboard/
    ├── clothes/
    ├── green-glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    ├── shoes/
    ├── trash/
    └── white-glass/

Directory Structure (After):
    data/
    ├── train/
    │   ├── battery/
    │   ├── biological/
    │   └── ...
    └── val/
        ├── battery/
        ├── biological/
        └── ...

Author: Azan Ahmed
"""

import os       # For directory and file path operations
import random   # For shuffling files to ensure a random split
import shutil   # For copying files from source to destination

# ──────────────────────────────────────────────────────────────
# Configuration: Define source and target directory paths
# ──────────────────────────────────────────────────────────────

source_dir = 'data set'         # Root folder containing class sub-folders of images
target_dir = "data/train"       # Destination folder for training images
val_dir = "data/val"            # Destination folder for validation images

# Train/validation split ratio: 80% training, 20% validation
split_ratio = 0.8

# ──────────────────────────────────────────────────────────────
# Create the top-level train and validation directories
# exist_ok=True prevents errors if the directories already exist
# ──────────────────────────────────────────────────────────────

os.makedirs(target_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Iterate through each class directory (e.g., battery, plastic)
# and split its images into train and validation sets
# ──────────────────────────────────────────────────────────────

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)

    # Skip any non-directory files in the source folder
    if not os.path.isdir(class_path):
        continue

    # Create class-specific subdirectories inside train/ and val/
    # e.g., data/train/battery/ and data/val/battery/
    train_class_dir = os.path.join(target_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # Get list of all files (images) in the current class directory
    files = os.listdir(class_path)

    # Shuffle files randomly so the split is not biased by filename order
    random.shuffle(files)

    # Calculate the index at which to split the list
    # e.g., if 1000 images and split_ratio=0.8, split_index=800
    split_index = int(len(files) * split_ratio)

    # Split the files into training and validation subsets
    train_files = files[:split_index]       # First 80% → training
    val_files = files[split_index:]          # Remaining 20% → validation

    # Copy each training file from source to the train directory
    # shutil.copy2 preserves file metadata (timestamps, etc.)
    for file in train_files:
        src = os.path.join(class_path, file)
        dst = os.path.join(train_class_dir, file)
        shutil.copy2(src, dst)

    # Copy each validation file from source to the validation directory
    for file in val_files:
        src = os.path.join(class_path, file)
        dst = os.path.join(val_class_dir, file)
        shutil.copy2(src, dst)

    # Print summary for each class showing the train/val distribution
    print(f"{class_name}: {len(train_files)} train files, {len(val_files)} validation files")
