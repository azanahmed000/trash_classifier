# 🗑️ Trash Classifier — Deep Learning Image Classification

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Trained_on-Kaggle_T4_GPU-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com)

A deep learning project that classifies trash images into **12 categories** using **transfer learning** with a pre-trained **ResNet-18** model. The model achieves **~92% accuracy** across training, validation, and test sets.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Categories](#categories)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [How It Works](#how-it-works)
- [Technologies Used](#technologies-used)
- [Author](#author)

---

## 🔍 Overview

Proper waste segregation is crucial for effective recycling and environmental sustainability. This project uses **computer vision** and **deep learning** to automatically classify images of trash into one of 12 categories, enabling smarter waste management systems.

The model leverages **transfer learning** — a technique where a model pre-trained on a large dataset (ImageNet, 1.2M images) is fine-tuned on our specific trash classification task. This allows us to achieve high accuracy even with a relatively smaller dataset.

---

## 🏷️ Categories

The classifier distinguishes between **12 types of waste**:

| # | Category | # | Category |
|---|----------|---|----------|
| 1 | 🔋 Battery | 7 | 🔩 Metal |
| 2 | 🌿 Biological | 8 | 📄 Paper |
| 3 | 🟤 Brown Glass | 9 | 🧴 Plastic |
| 4 | 📦 Cardboard | 10 | 👟 Shoes |
| 5 | 👕 Clothes | 11 | 🗑️ Trash |
| 6 | 🟢 Green Glass | 12 | ⚪ White Glass |

---

## 📊 Dataset

- **Total Images**: ~15,000+
- **Split Ratio**: 80% Train / 10% Validation / 10% Test
- **Preprocessing**: Images are resized to **224×224** pixels to match ResNet-18's expected input size
- **Augmentation** (training only):
  - Random horizontal flip
  - Random rotation (±10°)
  - Color jitter (brightness & contrast)
- **Normalization**: ImageNet mean & std values `([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`

---

## 🧠 Model Architecture

```
ResNet-18 (Pre-trained on ImageNet)
├── Conv Layers (FROZEN) — Feature extraction
│   ├── conv1 → bn1 → relu → maxpool
│   ├── layer1 (2 BasicBlocks)
│   ├── layer2 (2 BasicBlocks)
│   ├── layer3 (2 BasicBlocks)
│   └── layer4 (2 BasicBlocks)
├── AdaptiveAvgPool2d → 512-dim feature vector
└── FC Layer (TRAINABLE) — 512 → 12 classes
```

**Transfer Learning Strategy:**
- All convolutional layers are **frozen** (no weight updates)
- Only the final **fully-connected layer** is trained
- This approach is fast, efficient, and prevents overfitting

**Training Details:**
| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch Size | 64 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Weight Decay | 1e-4 |
| Loss Function | CrossEntropyLoss |

---

## 📈 Results

| Metric | Accuracy |
|--------|----------|
| **Training Accuracy** | 92.34% |
| **Validation Accuracy** | 92.08% |
| **Test Accuracy** | 92.14% |

The consistent accuracy across all three sets indicates the model generalizes well without overfitting.

| Epoch | Training Loss | Learning Rate |
|-------|---------------|---------------|
| 1 | 0.2699 | 0.001000 |
| 2 | 0.2608 | 0.001000 |
| 3 | 0.2505 | 0.001000 |
| 4 | 0.2488 | 0.001000 |
| 5 | 0.2361 | 0.001000 |

---

## 📁 Project Structure

```
trash-classifier/
├── preprocessing.ipynb        # Jupyter notebook — dataset splitting
├── preprocessing.py           # Commented Python script — dataset splitting
├── trash-calssifierier (2).ipynb  # Jupyter notebook — model training (Kaggle)
├── train.py                   # Commented Python script — model training
├── trash_classifier_resnet18.pth  # Saved model weights (not tracked by git)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

---

## 🚀 Setup & Usage

### Prerequisites

- Python 3.12+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/azanahmed000/trash_classifier.git
cd trash_classifier

# Install dependencies
pip install -r requirements.txt
```

### Running the Preprocessing Script

```bash
# Split raw images into train/val folders (80/20 split)
python preprocessing.py
```

### Training the Model

The model was trained on **Kaggle** using a **Tesla T4 GPU**. To retrain:

1. Upload the dataset to Kaggle or Google Drive
2. Open `trash-calssifierier (2).ipynb` in Kaggle
3. Run all cells

Or use the Python script locally (GPU recommended):

```bash
python train.py
```

### Loading the Trained Model (for Inference)

```python
import torch
from torchvision import models
import torch.nn as nn

# Rebuild the model architecture
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 12)

# Load the saved weights
model.load_state_dict(torch.load('trash_classifier_resnet18.pth'))
model.eval()
```

---

## ⚙️ How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────┐     ┌──────────────┐
│  Raw Images │ ──► │ Preprocessing│ ──► │ Data Augmentation   │ ──► │  ResNet-18   │
│  (12 classes)│     │ (80/20 split)│     │ (flip, rotate, etc.)│     │  (frozen CNN)│
└─────────────┘     └──────────────┘     └─────────────────────┘     └──────┬───────┘
                                                                            │
                                                                            ▼
                                                                    ┌──────────────┐
                                                                    │  FC Layer    │
                                                                    │  512 → 12    │
                                                                    │  (trainable) │
                                                                    └──────┬───────┘
                                                                            │
                                                                            ▼
                                                                    ┌──────────────┐
                                                                    │  Prediction  │
                                                                    │  (12 classes)│
                                                                    └──────────────┘
```

1. **Preprocessing** — Raw images are split into train (80%) and validation (20%) sets
2. **Augmentation** — Training images are randomly flipped, rotated, and color-adjusted
3. **Feature Extraction** — Frozen ResNet-18 CNN layers extract visual features
4. **Classification** — A trainable FC layer maps 512 features to 12 trash categories
5. **Evaluation** — Validation set is further split 50/50 into final validation and test sets

---

## 🛠️ Technologies Used

- **[PyTorch](https://pytorch.org/)** — Deep learning framework
- **[torchvision](https://pytorch.org/vision/)** — Pre-trained models & image transforms
- **[ResNet-18](https://arxiv.org/abs/1512.03385)** — Deep residual network architecture
- **[Kaggle](https://kaggle.com/)** — Cloud GPU training environment
- **[gdown](https://github.com/wkentaro/gdown)** — Google Drive file downloader

---

## 👤 Author

**Azan Ahmed**

- GitHub: [@azanahmed000](https://github.com/azanahmed000)

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
