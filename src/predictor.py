"""
Plant Disease Predictor - Model inference engine supporting both TFLite and Keras H5
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .utils import load_labels, parse_class_name, preprocess_image


class PlantDiseasePredictor:
    """
    Unified predictor supporting both TensorFlow Lite (.tflite) and Keras H5 (.h5).
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        labels_path: Optional[Union[str, Path]] = None,
        use_tflite: bool = True,
    ):
        base_dir = Path(__file__).resolve().parent.parent
        self.use_tflite = use_tflite

        # Resolve model path
        if model_path is None:
            if self.use_tflite:
                self.model_path = base_dir / "model" / "krishi_model.tflite"
            else:
                self.model_path = base_dir / "model" / "krishi_model.h5"
        else:
            self.model_path = Path(model_path)

        # Resolve labels path
        if labels_path is None:
            self.labels_path = base_dir / "model" / "labels.json"
        else:
            self.labels_path = Path(labels_path)

        self.labels = load_labels(self.labels_path)
        self.class_names = [item["raw"] for item in self.labels]
        self._init_model()

    def _init_model(self):
        """Initialize the model runtime based on file format."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        if self.use_tflite or str(self.model_path).endswith(".tflite"):
            self._init_tflite()
        else:
            self._init_keras()

    def _init_tflite(self):
        """Initialize TensorFlow Lite runtime (uses tflite-runtime or tensorflow)."""
        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
        except ImportError:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.is_tflite = True

    def _init_keras(self):
        """Initialize Keras H5 model runtime."""
        import tensorflow as tf
        self.model = tf.keras.models.load_model(str(self.model_path))
        self.is_tflite = False

    def predict(
        self,
        image_input,
        top_k: int = 3,
    ) -> Dict:
        """
        Run inference on an input image.

        Args:
            image_input: File path, bytes, or PIL Image.
            top_k: Number of highest-confidence predictions to return.

        Returns:
            Dictionary containing prediction summary and top-k probabilities.
        """
        tensor = preprocess_image(image_input)

        if self.is_tflite:
            self.interpreter.set_tensor(self.input_details[0]["index"], tensor)
            self.interpreter.invoke()
            probabilities = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        else:
            preds = self.model.predict(tensor, verbose=0)
            probabilities = preds[0]

        top_indices = np.argsort(probabilities)[::-1][:top_k]
        best_idx = int(top_indices[0])
        best_raw_class = self.class_names[best_idx]
        crop, disease, is_healthy = parse_class_name(best_raw_class)

        top_predictions = []
        for idx in top_indices:
            idx = int(idx)
            raw = self.class_names[idx]
            c, d, h = parse_class_name(raw)
            top_predictions.append({
                "class_id": idx,
                "raw_class": raw,
                "crop": c,
                "disease": d,
                "is_healthy": h,
                "confidence": float(probabilities[idx]),
            })

        return {
            "predicted_class": best_raw_class,
            "crop": crop,
            "disease": disease,
            "is_healthy": is_healthy,
            "confidence": float(probabilities[best_idx]),
            "top_predictions": top_predictions,
        }

