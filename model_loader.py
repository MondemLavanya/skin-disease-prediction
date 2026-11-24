"""
model_loader.py
==============================================================================
Skin Disease Prediction Using Deep Learning
B.Tech Minor Project-I - Model Loading & Weights Management Module
==============================================================================

Purpose:
--------
This module handles:
1. Discovering trained model weights in the 'model/' directory (.h5 or .keras)
2. Safe model loading with in-memory caching to avoid disk overhead per request
3. Graceful handling when model weights do not yet exist (pre-training stage)
4. Loading diagnostic class metadata (HAM10000 7-class benchmark)

Why Separate This Module?
-------------------------
In production and clean academic engineering:
- Decoupling model loading from prediction logic keeps code modular.
- If weights are missing, the Flask server still boots and serves documentation,
  dataset explorers, and preprocessing verification without throwing 500 errors.
==============================================================================
"""

import os
import json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_INFO_FILE = MODEL_DIR / "model_info.json"

# Potential model filenames
CANDIDATE_MODEL_FILES = [
    MODEL_DIR / "skin_disease_model.h5",
    MODEL_DIR / "skin_disease_model.keras",
    MODEL_DIR / "best_model.h5",
    MODEL_DIR / "best_model.keras"
]

# Global cache to keep loaded model in RAM across requests
_CACHED_MODEL = None
_CACHED_MODEL_PATH = None


def load_model_metadata():
    """
    Reads the model metadata JSON file containing class descriptions and configuration.
    
    Returns:
        dict: Parsed metadata or a default fallback if file cannot be read.
    """
    if MODEL_INFO_FILE.exists():
        try:
            with open(MODEL_INFO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to read {MODEL_INFO_FILE}: {e}")
            
    # Default fallback metadata for HAM10000 7-class benchmark
    return {
        "classes": [
            {"code": "akiec", "name": "Actinic Keratoses and Intraepithelial Carcinoma", "type": "Pre-cancerous"},
            {"code": "bcc", "name": "Basal Cell Carcinoma", "type": "Malignant"},
            {"code": "bkl", "name": "Benign Keratosis-like Lesions", "type": "Benign"},
            {"code": "df", "name": "Dermatofibroma", "type": "Benign"},
            {"code": "mel", "name": "Melanoma", "type": "Malignant (High Risk)"},
            {"code": "nv", "name": "Melanocytic Nevi", "type": "Benign"},
            {"code": "vasc", "name": "Vascular Lesions", "type": "Benign"}
        ]
    }


def find_model_file():
    """
    Scans the model/ directory for any valid saved Keras/TensorFlow weights file.
    
    Returns:
        Path or None: Path to the discovered weights file, or None if not trained yet.
    """
    for candidate in CANDIDATE_MODEL_FILES:
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 1024:
            return candidate
    return None


def get_model_status():
    """
    Checks whether a trained model weights file is available.
    
    Returns:
        dict: Status report containing:
            - is_loaded (bool)
            - model_path (str or None)
            - message (str)
            - classes (list[dict])
    """
    model_path = find_model_file()
    metadata = load_model_metadata()
    
    if model_path is not None:
        return {
            "is_loaded": True,
            "weights_exist": True,
            "model_path": str(model_path),
            "model_filename": model_path.name,
            "message": f"Trained model weights detected ({model_path.name}). Real inference is active.",
            "classes": metadata.get("classes", [])
        }
    else:
        return {
            "is_loaded": False,
            "weights_exist": False,
            "model_path": None,
            "model_filename": None,
            "message": "Model weights pending training (model/skin_disease_model.h5 not found). "
                       "Preprocessing and image validation are fully operational.",
            "classes": metadata.get("classes", [])
        }


def load_trained_model(force_reload=False):
    """
    Loads and returns the compiled Keras model.
    Caches the model in memory so subsequent predictions do not incur load latency.
    
    Parameters:
        force_reload (bool): If True, reloads the model from disk even if cached.
        
    Returns:
        tf.keras.Model or None: Loaded model, or None if no weights file exists.
    """
    global _CACHED_MODEL, _CACHED_MODEL_PATH
    
    # Return cached model if already in RAM and not forcing reload
    if _CACHED_MODEL is not None and not force_reload:
        return _CACHED_MODEL
    
    model_path = find_model_file()
    if model_path is None:
        # Academic integrity: Do not raise or fake, return None
        return None
    
    try:
        import tensorflow as tf
        print(f"[Model Loader] Loading model weights from: {model_path}")
        model = tf.keras.models.load_model(str(model_path))
        _CACHED_MODEL = model
        _CACHED_MODEL_PATH = str(model_path)
        print("[Model Loader] Model successfully loaded and cached in memory.")
        return _CACHED_MODEL
    except Exception as e:
        print(f"[Model Loader Error] Failed to load model weights from {model_path}: {e}")
        return None


def get_class_labels():
    """
    Returns the list of 7 class dictionaries with short codes, full clinical names, and severity types.
    """
    meta = load_model_metadata()
    return meta.get("classes", [])


def get_class_names():
    """
    Returns the sorted list of 7 class code strings:
    ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    """
    classes = get_class_labels()
    return [c["code"] for c in classes]


if __name__ == "__main__":
    print("=" * 60)
    print("Testing model_loader.py")
    status = get_model_status()
    print("Model Status:", json.dumps(status, indent=2))
    print("=" * 60)
