"""
Command-line inference script for NexAura Plant Disease Detection
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predictor import PlantDiseasePredictor


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using the NexAura plant disease detection model"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input leaf image (JPEG, PNG, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional path to model file (.tflite or .h5). Defaults to model/krishi_model.tflite",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to display (default: 3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format",
    )

    args = parser.parse_args()

    use_tflite = True
    if args.model and args.model.endswith(".h5"):
        use_tflite = False

    predictor = PlantDiseasePredictor(model_path=args.model, use_tflite=use_tflite)
    result = predictor.predict(args.image, top_k=args.top_k)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("\n" + "=" * 50)
    print("        NEXAURA PLANT DISEASE INFERENCE")
    print("=" * 50)
    print(f"Input Image : {args.image}")
    print(f"Crop        : {result['crop']}")
    print(f"Diagnosis   : {result['disease']}")
    print(f"Health State: {'Healthy' if result['is_healthy'] else 'Infected / Diseased'}")
    print(f"Confidence  : {result['confidence']:.2%}")
    print("-" * 50)
    print(f"Top {args.top_k} Predictions:")
    for i, pred in enumerate(result["top_predictions"], 1):
        print(f"  {i}. {pred['crop']} - {pred['disease']} ({pred['confidence']:.2%})")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
