"""
Verification script for NexAura model files and inference readiness
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predictor import PlantDiseasePredictor
from src.utils import load_labels


def run_checks():
    base_dir = Path(__file__).resolve().parent.parent
    model_dir = base_dir / "model"

    print("Running NexAura Model Verification...")
    print("-" * 50)

    # 1. File checks
    h5_path = model_dir / "krishi_model.h5"
    tflite_path = model_dir / "krishi_model.tflite"
    labels_path = model_dir / "labels.json"
    metadata_path = model_dir / "metadata.json"
    names_path = model_dir / "class_names.txt"

    for p in [h5_path, tflite_path, labels_path, metadata_path, names_path]:
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"[PASS] Found {p.name} ({size_mb:.2f} MB)")
        else:
            print(f"[FAIL] Missing {p.name}")
            sys.exit(1)

    # 2. Labels verification
    labels = load_labels(labels_path)
    print(f"[PASS] Successfully parsed {len(labels)} classes from labels.json")
    if len(labels) != 38:
        print(f"[FAIL] Expected 38 classes, found {len(labels)}")
        sys.exit(1)

    # 3. TFLite forward pass verification
    print("Testing TFLite model forward pass with dummy tensor...")
    predictor = PlantDiseasePredictor(model_path=tflite_path, use_tflite=True)
    
    # Generate dummy image (PIL RGB image of size 224x224)
    from PIL import Image
    dummy_img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
    
    result = predictor.predict(dummy_img, top_k=3)
    print(f"[PASS] TFLite Inference Successful!")
    print(f"       Sample Predicted Class : {result['predicted_class']}")
    print(f"       Confidence             : {result['confidence']:.2%}")
    print(f"       Top 3 Predictions Count: {len(result['top_predictions'])}")

    print("-" * 50)
    print("All Model Checks Passed Successfully!")


if __name__ == "__main__":
    run_checks()
