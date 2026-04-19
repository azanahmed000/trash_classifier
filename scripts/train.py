"""
train.py
=========
Trash Classifier — Model Training, Evaluation & Saving

This script trains a ResNet-18 model (pre-trained on ImageNet) using
transfer learning to classify trash images into 12 categories:
    battery, biological, brown-glass, cardboard, clothes, green-glass,
    metal, paper, plastic, shoes, trash, white-glass

Strategy:
    1. Load a pre-trained ResNet-18 from torchvision.
    2. Freeze all convolutional layers (feature extractor).
    3. Replace the final fully-connected layer with a new 12-class head.
    4. Train only the new FC layer for 5 epochs.
    5. Evaluate on validation and test sets.
    6. Save the trained model weights to disk.

Hardware:
    Originally trained on Kaggle with an NVIDIA Tesla T4 GPU.

Author: Azan Ahmed
"""

# ──────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────

import torch                             # Core PyTorch library
from torchvision import transforms       # Image transformations (resize, augment, normalize)
from torchvision import datasets         # ImageFolder dataset loader
from torchvision import models           # Pre-trained model zoo (ResNet, VGG, etc.)
from torch.utils.data import DataLoader  # Batched data loading with shuffling
from torch.utils.data import random_split  # Splitting a dataset into subsets
import torch.nn as nn                    # Neural network layers and loss functions

# ──────────────────────────────────────────────────────────────
# Step 1: Download and Extract the Dataset (Kaggle-specific)
# ──────────────────────────────────────────────────────────────
# NOTE: The lines below are designed for running inside a Kaggle notebook.
#       They download the dataset zip from Google Drive using gdown and
#       extract it. If running locally, place your data/ folder manually.
#
# !pip install gdown
# import gdown
#
# file_id = '1UrTmhu2nLvWMv-AMfoa3Z6JBZwalagA7'
# url = f'https://drive.google.com/uc?id={file_id}'
# output = 'data.zip'
# gdown.download(url, output, quiet=False)
#
# !unzip -q /kaggle/working/data.zip -d /kaggle/working/my_data

# ──────────────────────────────────────────────────────────────
# Step 2: Define Dataset Paths
# ──────────────────────────────────────────────────────────────
# These paths point to the extracted train/ and val/ directories.
# Update these if running on a local machine.

train_path = "/kaggle/working/my_data/content/drive/MyDrive/datasets/data/train"
val_path   = "/kaggle/working/my_data/content/drive/MyDrive/datasets/data/val"

# ──────────────────────────────────────────────────────────────
# Step 3: Define Image Transforms (Augmentation + Normalization)
# ──────────────────────────────────────────────────────────────

# Training transforms include data augmentation to improve generalization:
#   - Resize all images to 224×224 (required by ResNet-18)
#   - RandomHorizontalFlip: randomly mirror images horizontally (50% chance)
#   - RandomRotation: rotate images by up to ±10 degrees
#   - ColorJitter: randomly adjust brightness and contrast
#   - ToTensor: convert PIL image → PyTorch tensor (HWC → CHW, scale to [0,1])
#   - Normalize: standardize using ImageNet channel means and standard deviations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet channel means
        std=[0.229, 0.224, 0.225]      # ImageNet channel standard deviations
    )
])

# Validation/Test transforms — NO augmentation, only resize + normalize
# We want deterministic, unmodified images for fair evaluation
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet channel means
        std=[0.229, 0.224, 0.225]      # ImageNet channel standard deviations
    )
])

# ──────────────────────────────────────────────────────────────
# Step 4: Load Datasets Using ImageFolder
# ──────────────────────────────────────────────────────────────
# ImageFolder expects the directory structure:
#   train_path/class_name/image_file.jpg
# It automatically assigns integer labels based on sorted folder names.

train_data = datasets.ImageFolder(train_path, transform=train_transforms)
val_data   = datasets.ImageFolder(val_path, transform=val_test_transforms)

# ──────────────────────────────────────────────────────────────
# Step 5: Create DataLoaders (Train / Validation / Test)
# ──────────────────────────────────────────────────────────────

# Training DataLoader — shuffle=True ensures random batch composition each epoch
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# The physical 'val' folder is further split into validation and test sets
# using a 50/50 random split with a fixed seed for reproducibility
total_val_images = len(val_data)
test_size = total_val_images // 2           # Half for testing
new_val_size = total_val_images - test_size # Remaining half for validation

# Fixed random seed (42) ensures the same split every time
generator = torch.Generator().manual_seed(42)
new_val_data, test_data = random_split(val_data, [new_val_size, test_size], generator=generator)

# Validation and Test DataLoaders — shuffle=False for consistent evaluation
val_loader  = DataLoader(new_val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print(f"Total Images in physical 'val' folder: {total_val_images}")
print(f"--> Assigned to Validation Loader: {len(new_val_data)}")
print(f"--> Assigned to Test Loader: {len(test_data)}")

# ──────────────────────────────────────────────────────────────
# Step 6: Load Pre-trained ResNet-18 and Modify for 12-Class Output
# ──────────────────────────────────────────────────────────────
# ResNet-18 was originally trained on ImageNet (1000 classes).
# We replace the final fully-connected (FC) layer to output 12 classes
# matching our 12 trash categories.

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 12)   # in_features=512 → 12 classes

# ──────────────────────────────────────────────────────────────
# Step 7: Freeze All Layers Except the Final FC Layer
# ──────────────────────────────────────────────────────────────
# Transfer learning strategy: keep the convolutional feature extractor
# frozen (no gradient updates) and only train the new classification head.
# This is efficient and prevents overfitting on smaller datasets.

for param in model.parameters():
    param.requires_grad = False         # Freeze all layers

for param in model.fc.parameters():
    param.requires_grad = True          # Unfreeze only the final FC layer

# ──────────────────────────────────────────────────────────────
# Step 8: Define Loss Function and Optimizer
# ──────────────────────────────────────────────────────────────
# CrossEntropyLoss: standard loss for multi-class classification
#   - Internally applies softmax + negative log-likelihood
# Adam optimizer: adaptive learning rate optimizer
#   - lr=0.001: initial learning rate
#   - weight_decay=1e-4: L2 regularization to prevent overfitting

loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-4)

# ──────────────────────────────────────────────────────────────
# Step 9: Training Loop — 5 Epochs
# ──────────────────────────────────────────────────────────────

for epoch in range(5):
    model.train()       # Set model to training mode (enables dropout, batchnorm updates)
    total_loss = 0      # Accumulator for epoch loss

    for (images, labels) in train_loader:
        # Forward pass: compute model predictions
        outputs = model(images)

        # Compute the cross-entropy loss between predictions and true labels
        loss = loss_fn(outputs, labels)

        # Backward pass: zero gradients → compute gradients → update weights
        optimizer.zero_grad()   # Clear gradients from previous step
        loss.backward()         # Backpropagate the loss
        optimizer.step()        # Update the FC layer weights

        total_loss += loss.item()   # Accumulate batch loss

    # Log the average loss and current learning rate for this epoch
    current_lr = optimizer.param_groups[0]['lr']
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/5, Loss: {avg_loss:.4f}, Learning Rate = {current_lr:.6f}")

# ──────────────────────────────────────────────────────────────
# Step 10: Evaluate on Training Set
# ──────────────────────────────────────────────────────────────

train_correct = 0   # Number of correctly classified training images
train_total = 0     # Total number of training images evaluated

for images, labels in train_loader:
    outputs = model(images)
    _, preds = torch.max(outputs, 1)    # Get the class with highest score

    train_correct += (preds == labels).sum().item()
    train_total += labels.size(0)

train_acc = train_correct / train_total
print(f"Train Accuracy: {train_acc * 100:.2f}%")

# ──────────────────────────────────────────────────────────────
# Step 11: Evaluate on Validation Set
# ──────────────────────────────────────────────────────────────

model.eval()        # Set model to evaluation mode (disables dropout, freezes batchnorm)
correct = 0
total = 0

with torch.no_grad():       # Disable gradient computation for faster inference
    for (images, labels) in val_loader:
        output = model(images)
        _, pred = torch.max(output, 1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")

# ──────────────────────────────────────────────────────────────
# Step 12: Evaluate on Test Set
# ──────────────────────────────────────────────────────────────

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for (images, labels) in test_loader:
        outputs = model(images)
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# ──────────────────────────────────────────────────────────────
# Step 13: Save the Trained Model Weights
# ──────────────────────────────────────────────────────────────
# Save only the state_dict (model parameters), not the full model object.
# This is the recommended approach for portability and flexibility.

torch.save(model.state_dict(), 'trash_classifier_resnet18.pth')
print("Model saved successfully!")
