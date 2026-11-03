"""
Utility functions for preprocessing and label resolution
"""

import io
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
from PIL import Image


def load_labels(labels_path: Union[str, Path] = "model/labels.json") -> List[Dict]:
    """Load labels configuration file."""
    path = Path(labels_path)
    if not path.exists():
        # Fallback to search relative to current file
        path = Path(__file__).resolve().parent.parent / "model" / "labels.json"
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_class_name(raw: str) -> Tuple[str, str, bool]:
    """
    Parse a PlantVillage formatted class string into clean metadata.
    
    Example:
        'Tomato___Late_blight' -> ('Tomato', 'Late Blight', False)
        'Apple___healthy'      -> ('Apple', 'Healthy', True)
    """
    parts = raw.split("___")
    crop = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").replace("  ", " ").strip().title() if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in disease.lower()
    return crop, disease, is_healthy


def preprocess_image(image_input: Union[str, Path, bytes, Image.Image], target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess input image to model input tensor specification:
    - Resize to target_size (default: 224x224)
    - Convert to RGB
    - Rescale pixel values [0, 1] (float32)
    - Add batch dimension: shape (1, 224, 224, 3)
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    img = img.convert("RGB")
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)
