"""
Quickstart Example - How to load and predict in 5 lines of code
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predictor import PlantDiseasePredictor


def main():
    # 1. Initialize predictor (automatically selects model/krishi_model.tflite)
    predictor = PlantDiseasePredictor()

    # 2. For demonstration, generate a synthetic RGB leaf image (replace with your file path: "leaf.jpg")
    synthetic_leaf = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))

    # 3. Predict disease
    prediction = predictor.predict(synthetic_leaf, top_k=3)

    print("Prediction Results:")
    print("Crop:", prediction["crop"])
    print("Disease:", prediction["disease"])
    print("Is Healthy?:", prediction["is_healthy"])
    print(f"Confidence: {prediction['confidence']:.2%}")


if __name__ == "__main__":
    main()
