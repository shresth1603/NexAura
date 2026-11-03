# NexAura: Plant Disease Detection Deep Learning Model

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://tensorflow.org/)
[![TFLite](https://img.shields.io/badge/TFLite-Edge%20%26%20Mobile-green.svg)](https://www.tensorflow.org/lite)
[![Classes](https://img.shields.io/badge/Classes-38%20Plant%20Conditions-brightgreen.svg)](#38-supported-plant-disease-classes)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**NexAura** provides production-ready, lightweight deep learning models engineered for automated crop leaf disease diagnosis and health monitoring. Powered by MobileNetV2 transfer learning trained on the benchmark PlantVillage dataset, it classifies **38 distinct crop disease and healthy states** across **14 agricultural crops**.

---

## 📂 Repository Structure

The repository is systematically organized for clarity and frictionless integration into web backends, mobile apps, and edge devices:

```text
NexAura/
├── model/
│   ├── krishi_model.h5          # Full Keras model (Weights & Architecture, ~9.7 MB)
│   ├── krishi_model.tflite      # TensorFlow Lite model for mobile/edge/server (~8.6 MB)
│   ├── labels.json              # Complete class mapping (id, crop, disease, healthy flag)
│   ├── metadata.json            # Model architecture, tensor shapes, and preprocessing specs
│   └── class_names.txt          # Plain-text newline-separated list of all 38 classes
├── src/
│   ├── __init__.py              # Package entry point
│   ├── predictor.py             # Unified inference engine (auto-selects TFLite or Keras H5)
│   └── utils.py                 # Image preprocessing and label parsing utilities
├── scripts/
│   ├── predict.py               # CLI tool: python scripts/predict.py --image <path>
│   └── verify_models.py         # Automated verification & dummy inference sanity check
├── examples/
│   └── quickstart.py            # Minimal 5-line sample script
├── .gitignore                   # Ignore rules for pycache, checkpoints, and venvs
├── requirements.txt             # Minimal dependencies for inference
└── README.md                    # Comprehensive documentation and integration guide
```

---

## 🔬 Model Specifications

| Parameter | Specification |
| :--- | :--- |
| **Base Architecture** | MobileNetV2 (Transfer Learning from ImageNet) |
| **Input Tensor Shape** | `(1, 224, 224, 3)` (Batch, Height, Width, Channels) |
| **Color Space** | RGB |
| **Normalization** | Pixel values scaled to `[0.0, 1.0]` (`pixel / 255.0`) |
| **Output Shape** | `(1, 38)` Softmax probabilities |
| **Classes** | 38 plant disease & healthy categories |
| **Dataset Source** | [PlantVillage Dataset](https://www.tensorflow.org/datasets/catalog/plant_village) (54,000+ images) |
| **TFLite Model Size** | **8.63 MB** (ideal for mobile, edge, and low-latency APIs) |
| **Keras H5 Model Size**| **9.68 MB** (ideal for Python backends & fine-tuning) |

---

## 🚀 Quick Start Guide

### 1. Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/shresth1603/NexAura.git
cd NexAura
pip install -r requirements.txt
```

### 2. Verify Models

Run the automated verification script to confirm model weights, label integrity, and test inference:

```bash
python scripts/verify_models.py
```

### 3. Run Inference from CLI

Test any leaf photograph using the CLI tool:

```bash
python scripts/predict.py --image path/to/leaf.jpg
```

**Output Example:**
```text
==================================================
        NEXAURA PLANT DISEASE INFERENCE
==================================================
Input Image : sample_leaf.jpg
Crop        : Tomato
Diagnosis   : Early Blight
Health State: Infected / Diseased
Confidence  : 96.42%
--------------------------------------------------
Top 3 Predictions:
  1. Tomato - Early Blight (96.42%)
  2. Tomato - Target Spot (2.15%)
  3. Tomato - Septoria Leaf Spot (0.83%)
==================================================
```

For JSON output:
```bash
python scripts/predict.py --image path/to/leaf.jpg --json
```

---

## 💻 Python Code Integration

### Using the Built-In Predictor

```python
from src.predictor import PlantDiseasePredictor

# Automatically loads model/krishi_model.tflite (fastest)
predictor = PlantDiseasePredictor()

# Pass image file path, bytes, or PIL Image
result = predictor.predict("leaf.jpg", top_k=3)

print("Crop:", result["crop"])
print("Diagnosis:", result["disease"])
print("Is Healthy:", result["is_healthy"])
print(f"Confidence: {result['confidence']:.2%}")
```

### Using Pure TensorFlow Lite (Zero Custom Wrapper)

```python
import numpy as np
from PIL import Image
import tensorflow as tf

# 1. Load model
interpreter = tf.lite.Interpreter(model_path="model/krishi_model.tflite")
interpreter.allocate_tensors()

input_idx = interpreter.get_input_details()[0]["index"]
output_idx = interpreter.get_output_details()[0]["index"]

# 2. Preprocess image (224x224 RGB, scaled 0-1)
image = Image.open("leaf.jpg").convert("RGB").resize((224, 224))
input_data = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)

# 3. Predict
interpreter.set_tensor(input_idx, input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_idx)[0]

predicted_id = int(np.argmax(output))
confidence = float(np.max(output))

print(f"Predicted Class ID: {predicted_id} ({confidence:.2%})")
```

---

## 📱 Mobile & Edge Deployment (Android / iOS / Flutter)

The `model/krishi_model.tflite` file is completely standalone and optimized for mobile apps.

### Android (Kotlin) Snippet
```kotlin
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer

val tfliteModel = loadModelFile(context, "krishi_model.tflite")
val interpreter = Interpreter(tfliteModel)

// Input: ByteBuffer of size 1 * 224 * 224 * 3 * 4 (Float32)
// Output: Array of shape [1][38]
val output = Array(1) { FloatArray(38) }
interpreter.run(inputBuffer, output)
```

---

## 🌿 38 Supported Plant Disease Classes

The model classifies leaves into the following 38 categories:

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

## 📦 Compatibility & Direct Access

Both models are strictly below GitHub's 100 MB file size limit (~9 MB each) and require **no Git LFS bandwidth tokens or complex credentials** to download. You can clone the repo directly or download single files via raw URLs:

- **TFLite Model Direct Link:**  
  `https://raw.githubusercontent.com/shresth1603/NexAura/main/model/krishi_model.tflite`
- **Labels JSON Direct Link:**  
  `https://raw.githubusercontent.com/shresth1603/NexAura/main/model/labels.json`

---

## 📄 License
This repository is licensed under the MIT License.
