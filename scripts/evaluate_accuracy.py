"""
NexAura - Model Prediction Accuracy & Latency Benchmark Script
=============================================================
Evaluates Top-1 and Top-3 accuracy, confidence scores, and inference latency
across test datasets or curated sample folders.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predictor import PlantDiseasePredictor


def run_benchmark(data_dir: str, samples_per_class: int = 50, use_tflite: bool = True):
    base_path = Path(data_dir)
    if not base_path.exists():
        print(f"Error: Data directory not found at {base_path}")
        sys.exit(1)

    print("=" * 75)
    print("        NEXAURA: MODEL ACCURACY & PERFORMANCE EVALUATION")
    print("=" * 75)
    print(f"Target Directory   : {base_path.resolve()}")
    print(f"Engine             : {'TensorFlow Lite (TFLite)' if use_tflite else 'Keras H5'}")
    print(f"Max Samples/Class  : {samples_per_class}")
    print("-" * 75)

    predictor = PlantDiseasePredictor(use_tflite=use_tflite)

    # Detect class subdirectories
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not subdirs:
        print("No class subdirectories found in target folder.")
        sys.exit(1)

    results: Dict[str, Dict] = {}
    total_top1 = 0
    total_top3 = 0
    total_samples = 0
    all_latencies: List[float] = []
    misclassifications: List[Dict] = []

    for d in sorted(subdirs, key=lambda x: x.name):
        class_name = d.name
        images = list(d.glob("*.JPG")) + list(d.glob("*.jpg")) + list(d.glob("*.png")) + list(d.glob("*.jpeg"))
        if not images:
            continue

        selected_images = images[:samples_per_class]
        results[class_name] = {
            "total": len(selected_images),
            "top1": 0,
            "top3": 0,
            "confidences": [],
            "latencies": [],
        }

        for img in selected_images:
            t0 = time.perf_counter()
            pred = predictor.predict(str(img), top_k=3)
            dt = (time.perf_counter() - t0) * 1000.0

            all_latencies.append(dt)
            results[class_name]["latencies"].append(dt)
            results[class_name]["confidences"].append(pred["confidence"])
            total_samples += 1

            if pred["predicted_class"] == class_name:
                results[class_name]["top1"] += 1
                total_top1 += 1
            else:
                misclassifications.append({
                    "true_class": class_name,
                    "pred_class": pred["predicted_class"],
                    "confidence": pred["confidence"],
                    "file": img.name,
                })

            top3 = [p["raw_class"] for p in pred["top_predictions"]]
            if class_name in top3:
                results[class_name]["top3"] += 1
                total_top3 += 1

    # Print summary table
    print(f"{'Class Name':<35} | {'Top-1 Acc':<10} | {'Top-3 Acc':<10} | {'Avg Conf':<10} | {'Latency':<8}")
    print("-" * 75)

    for c, r in results.items():
        top1_acc = (r["top1"] / r["total"]) * 100 if r["total"] > 0 else 0
        top3_acc = (r["top3"] / r["total"]) * 100 if r["total"] > 0 else 0
        avg_conf = np.mean(r["confidences"]) * 100 if r["confidences"] else 0
        avg_lat = np.mean(r["latencies"]) if r["latencies"] else 0
        print(f"{c:<35} | {top1_acc:>8.1f}% | {top3_acc:>8.1f}% | {avg_conf:>8.1f}% | {avg_lat:>6.1f} ms")

    overall_top1 = (total_top1 / total_samples) * 100 if total_samples > 0 else 0
    overall_top3 = (total_top3 / total_samples) * 100 if total_samples > 0 else 0
    avg_total_lat = np.mean(all_latencies) if all_latencies else 0
    avg_total_conf = np.mean([c for r in results.values() for c in r["confidences"]]) * 100 if total_samples > 0 else 0

    print("=" * 75)
    print(f"BENCHMARK SUMMARY ({total_samples} samples across {len(results)} classes):")
    print(f"  • OVERALL TOP-1 ACCURACY : {overall_top1:.2f}% ({total_top1}/{total_samples})")
    print(f"  • OVERALL TOP-3 ACCURACY : {overall_top3:.2f}% ({total_top3}/{total_samples})")
    print(f"  • MEAN CONFIDENCE SCORE  : {avg_total_conf:.2f}%")
    print(f"  • MEAN CPU LATENCY/IMAGE : {avg_total_lat:.2f} ms")
    print("=" * 75)

    if misclassifications:
        print(f"\nMisclassification Diagnostics (Sample of {min(5, len(misclassifications))}):")
        for m in misclassifications[:5]:
            print(f"  - Actual: {m['true_class']}")
            print(f"    Predicted: {m['pred_class']} ({m['confidence']:.1%}) [{m['file']}]")


def main():
    parser = argparse.ArgumentParser(description="NexAura Model Benchmark Suite")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing class subfolders of leaf images")
    parser.add_argument("--samples", type=int, default=50, help="Number of images per class to test (default: 50)")
    parser.add_argument("--use_keras", action="store_true", help="Use Keras H5 engine instead of TFLite")
    args = parser.parse_args()

    run_benchmark(data_dir=args.data_dir, samples_per_class=args.samples, use_tflite=not args.use_keras)


if __name__ == "__main__":
    main()
