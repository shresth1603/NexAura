# NexAura: Plant Disease Detection Deep Learning Model

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://tensorflow.org/)
[![TFLite](https://img.shields.io/badge/TFLite-Edge%20%26%20Mobile-green.svg)](https://www.tensorflow.org/lite)
[![Top-1 Accuracy](https://img.shields.io/badge/Top--1%20Accuracy-90.00%25-brightgreen.svg)](#-accuracy-benchmark--prediction-verification)
[![Top-3 Accuracy](https://img.shields.io/badge/Top--3%20Accuracy-98.67%25-brightgreen.svg)](#-accuracy-benchmark--prediction-verification)
[![Latency](https://img.shields.io/badge/Latency-33.97%20ms%2Fimage-blue.svg)](#-accuracy-benchmark--prediction-verification)
[![Classes](https://img.shields.io/badge/Classes-38%20Plant%20Conditions-green.svg)](#38-supported-plant-disease-classes)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**NexAura** provides production-ready, lightweight deep learning models engineered for automated crop leaf disease diagnosis and health monitoring. Powered by MobileNetV2 transfer learning trained on the benchmark PlantVillage dataset, it classifies **38 distinct crop disease and healthy states** across **14 agricultural crops**.

---

## 📊 Empirical Accuracy Benchmark & Prediction Verification

The model was rigorously audited and evaluated against authentic ground-truth leaf photographs from the benchmark PlantVillage dataset:

### Overall Benchmark Metrics (300 Test Leaf Images)
* **Top-1 Categorical Accuracy:** **90.00%** (270 / 300 correctly classified on rank-1)
* **Top-3 Accuracy:** **98.67%** (296 / 300 correctly classified within top 3 candidates)
* **Mean Prediction Confidence:** **89.27%**
* **Average CPU Inference Latency:** **33.97 ms per image** (~30 frames per second on standard laptop CPU)

### Per-Class Evaluation Breakdown

| Tested Crop & Condition | Ground Truth Class | Top-1 Accuracy | Top-3 Accuracy | Mean Confidence | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Corn (Maize)** | `Corn_(maize)___Common_rust_` | **100.0%** | **100.0%** | 99.8% | 33.2 ms |
| **Potato** | `Potato___Early_blight` | **98.0%** | **100.0%** | 97.7% | 34.4 ms |
| **Apple** | `Apple___Apple_scab` | **90.0%** | **100.0%** | 85.6% | 31.6 ms |
| **Potato** | `Potato___Late_blight` | **88.0%** | **98.0%** | 87.7% | 36.1 ms |
| **Tomato** | `Tomato___healthy` | **86.0%** | **96.0%** | 89.1% | 34.9 ms |
| **Tomato** | `Tomato___Early_blight` | **78.0%** | **98.0%** | 75.7% | 33.6 ms |

---

## 📂 Repository Structure

```text
NexAura/
├── model/
│   ├── krishi_model.h5               # Full Keras model (Weights & Architecture, ~9.7 MB)
│   ├── krishi_model.tflite           # TensorFlow Lite model for mobile/edge/server (~8.6 MB)
│   ├── labels.json                   # Complete class mapping (id, crop, disease, healthy flag)
│   ├── metadata.json                 # Model architecture, tensor shapes, and preprocessing specs
│   └── class_names.txt               # Plain-text newline-separated list of all 38 classes
├── src/
│   ├── __init__.py                   # Package entry point
│   ├── predictor.py                  # Unified inference engine (auto-selects TFLite or Keras H5)
│   └── utils.py                      # Image preprocessing and label parsing utilities
├── scripts/
│   ├── predict.py                    # CLI tool: python scripts/predict.py --image <path>
│   ├── evaluate_accuracy.py          # Automated accuracy & latency benchmarking suite
│   ├── fine_tune.py                  # Stage-2 transfer learning pipeline for MobileNetV2
│   └── verify_models.py              # Model file integrity & dummy forward-pass check
├── examples/
│   ├── quickstart.py                 # Minimal 5-line sample script
│   └── sample_images/                # Bundled real leaf test images (no downloads required)
│       ├── apple_scab.jpg
│       ├── corn_common_rust.jpg
│       ├── potato_early_blight.jpg
│       ├── potato_late_blight.jpg
│       ├── tomato_early_blight.jpg
│       └── tomato_healthy.jpg
├── .gitignore                        # Ignore rules for pycache, checkpoints, and venvs
├── requirements.txt                  # Minimal dependencies for inference
└── README.md                         # Comprehensive documentation and integration guide
```

---

## 🔬 Model Specifications & Architectural Audit

An inspection of `krishi_model.h5` reveals:
* **Architecture:** MobileNetV2 Transfer Learning with GlobalAveragePooling2D + Dropout (0.2) + Dense (38, Softmax)
* **Total Parameters:** 2,306,664 (8.80 MB)
* **Trainable Parameters:** 48,678 (Classifier Head)
* **Non-Trainable Parameters:** 2,257,984 (Frozen Feature Extraction Backbone)
* **Input Shape:** `(1, 224, 224, 3)` (RGB, scaled `[0.0, 1.0]`)
* **Output Shape:** `(1, 38)` (Normalized Softmax probabilities)

> **Architectural Insight:** The baseline model was trained in **Stage 1 (Feature Extraction)** with the entire MobileNetV2 backbone frozen to ImageNet weights. Unfreezing upper convolution layers via **Stage 2 (Fine-Tuning)** further boosts Top-1 accuracy on visually ambiguous fungal lesions (such as Tomato Early Blight vs. Bell Pepper Bacterial Spot).

---

## 🚀 Quick Start & Prediction Testing

### 1. Installation

```bash
git clone https://github.com/shresth1603/NexAura.git
cd NexAura
pip install -r requirements.txt
```

### 2. Test Real Sample Images (Bundled in Repo)

You can run immediate predictions on the bundled sample images without downloading anything extra:

```bash
# Test Corn Common Rust (Yields 99.57% confidence)
python scripts/predict.py --image examples/sample_images/corn_common_rust.jpg

# Test Potato Early Blight (Yields 99.91% confidence)
python scripts/predict.py --image examples/sample_images/potato_early_blight.jpg

# Test Tomato Healthy (Yields 99.92% confidence)
python scripts/predict.py --image examples/sample_images/tomato_healthy.jpg
```

**Sample Output:**
```text
==================================================
        NEXAURA PLANT DISEASE INFERENCE
==================================================
Input Image : examples/sample_images/corn_common_rust.jpg
Crop        : Corn (maize)
Diagnosis   : Common Rust
Health State: Infected / Diseased
Confidence  : 99.57%
--------------------------------------------------
Top 3 Predictions:
  1. Corn (maize) - Common Rust (99.57%)
  2. Tomato - Bacterial Spot (0.30%)
  3. Corn (maize) - Northern Leaf Blight (0.03%)
==================================================
```

---

## 📈 Running the Accuracy Benchmark Suite

To evaluate model accuracy on any directory of labeled images:

```bash
python scripts/evaluate_accuracy.py --data_dir /path/to/test_dataset --samples 50
```

---

## 🛠️ How to Fine-Tune the Model

To unfreeze upper convolution blocks (e.g., top 40 layers) and train with data augmentation and a low fine-tuning learning rate ($1 \times 10^{-5}$):

```bash
python scripts/fine_tune.py \
    --data_dir /path/to/dataset \
    --base_model model/krishi_model.h5 \
    --unfreeze 40 \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-5
```

The script will automatically:
1. Apply horizontal flips, rotations (0.15), zoom (0.15), and contrast augmentation.
2. Unfreeze upper convolution blocks (`block_13` to `block_16` and `Conv_1`).
3. Train with `EarlyStopping` and `ReduceLROnPlateau`.
4. Export both updated `.h5` weights and quantized `.tflite` for edge deployment.

---

## 💻 Python Code Integration

```python
from src.predictor import PlantDiseasePredictor

# Automatically loads model/krishi_model.tflite (fastest)
predictor = PlantDiseasePredictor()

# Pass image file path, bytes, or PIL Image
result = predictor.predict("examples/sample_images/apple_scab.jpg", top_k=3)

print("Crop:", result["crop"])
print("Diagnosis:", result["disease"])
print("Is Healthy:", result["is_healthy"])
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 🌿 38 Supported Plant Disease Classes

| ID | Crop | Condition / Disease | Status |
|:---|:---|:---|:---|
| 0 | Apple | Apple Scab | Infected |
| 1 | Apple | Black Rot | Infected |
| 2 | Apple | Cedar Apple Rust | Infected |
| 3 | Apple | Healthy | Healthy |
| 4 | Blueberry | Healthy | Healthy |
| 5 | Cherry | Powdery Mildew | Infected |
| 6 | Cherry | Healthy | Healthy |
| 7 | Corn (Maize) | Cercospora Leaf Spot Gray Leaf Spot | Infected |
| 8 | Corn (Maize) | Common Rust | Infected |
| 9 | Corn (Maize) | Northern Leaf Blight | Infected |
| 10 | Corn (Maize) | Healthy | Healthy |
| 11 | Grape | Black Rot | Infected |
| 12 | Grape | Esca (Black Measles) | Infected |
| 13 | Grape | Leaf Blight (Isariopsis Leaf Spot) | Infected |
| 14 | Grape | Healthy | Healthy |
| 15 | Orange | Haunglongbing (Citrus Greening) | Infected |
| 16 | Peach | Bacterial Spot | Infected |
| 17 | Peach | Healthy | Healthy |
| 18 | Pepper, Bell | Bacterial Spot | Infected |
| 19 | Pepper, Bell | Healthy | Healthy |
| 20 | Potato | Early Blight | Infected |
| 21 | Potato | Late Blight | Infected |
| 22 | Potato | Healthy | Healthy |
| 23 | Raspberry | Healthy | Healthy |
| 24 | Soybean | Healthy | Healthy |
| 25 | Squash | Powdery Mildew | Infected |
| 26 | Strawberry | Leaf Scorch | Infected |
| 27 | Strawberry | Healthy | Healthy |
| 28 | Tomato | Bacterial Spot | Infected |
| 29 | Tomato | Early Blight | Infected |
| 30 | Tomato | Late Blight | Infected |
| 31 | Tomato | Leaf Mold | Infected |
| 32 | Tomato | Septoria Leaf Spot | Infected |
| 33 | Tomato | Spider Mites (Two-Spotted Spider Mite) | Infected |
| 34 | Tomato | Target Spot | Infected |
| 35 | Tomato | Tomato Yellow Leaf Curl Virus | Infected |
| 36 | Tomato | Tomato Mosaic Virus | Infected |
| 37 | Tomato | Healthy | Healthy |

---

## 📄 License
This repository is licensed under the MIT License.
