"""
predictor.py
==============================================================================
Skin Disease Prediction Using Deep Learning
B.Tech Minor Project-I - Prediction & Inference Orchestrator
==============================================================================

Purpose:
--------
This module orchestrates:
1. File verification (safe extension, non-empty, readable by PIL)
2. Image preprocessing (resizing to 224x224, RGB conversion, [0, 1] normalization)
3. Tensor diagnostic extraction (tensor shape, pixel range, mean intensity)
4. Deep learning inference via model_loader
5. Academic integrity enforcement:
   - When trained weights exist: runs genuine softmax inference and returns real probabilities.
   - When weights are not yet trained: verifies preprocessing and returns an honest,
     informative pre-training diagnostic response rather than fake predictions.

Why This Approach?
------------------
In medical computer vision projects, fabricating diagnostic predictions or
hallucinating 98% accuracy without trained weights violates academic ethics.
This module verifies that the data pipeline is 100% ready for model weights.
==============================================================================
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image

# Import preprocessing utilities
from preprocess import (
    load_and_preprocess_single_image,
    validate_image_file,
    IMG_SIZE,
    IMG_SHAPE
)

# Import model loading utilities
from model_loader import (
    load_trained_model,
    get_model_status,
    get_class_labels,
    get_class_names
)


def extract_image_metadata(image_path):
    """
    Extracts raw image metadata (dimensions, format, file size) prior to preprocessing.
    
    Parameters:
        image_path (str or Path): Path to the image file.
        
    Returns:
        dict: Image properties.
    """
    path = Path(image_path)
    file_size_kb = round(path.stat().st_size / 1024, 2)
    
    with Image.open(path) as img:
        width, height = img.size
        img_format = img.format or path.suffix.upper().replace('.', '')
        mode = img.mode
        
    return {
        "filename": path.name,
        "width": width,
        "height": height,
        "dimensions": f"{width} × {height} px",
        "format": img_format,
        "mode": mode,
        "file_size_kb": file_size_kb
    }


def predict_skin_disease(image_path):
    """
    Main prediction pipeline:
    1. Validates image integrity and extension.
    2. Resizes to (224, 224) and normalizes pixel values to [0.0, 1.0].
    3. Checks if model weights are loaded:
       - If loaded: runs genuine model inference and calculates softmax confidence.
       - If pending: returns preprocessed tensor diagnostics and an academic status report.
       
    Parameters:
        image_path (str or Path): Filepath to the uploaded image.
        
    Returns:
        dict: Structured prediction or pre-training verification result.
    """
    path = Path(image_path)
    
    # Step 1: Validate file existence and integrity
    is_valid, error_msg = validate_image_file(path)
    if not is_valid:
        return {
            "status": "error",
            "error_type": "ValidationFailure",
            "message": f"Invalid or corrupted image: {error_msg}"
        }
        
    try:
        # Step 2: Extract raw file metadata
        raw_meta = extract_image_metadata(path)
        
        # Step 3: Preprocess image into normalized batch tensor (1, 224, 224, 3)
        tensor = load_and_preprocess_single_image(path, target_size=IMG_SIZE, normalize=True)
        
        # Calculate genuine numerical statistics of the preprocessed tensor
        tensor_min = float(np.min(tensor))
        tensor_max = float(np.max(tensor))
        tensor_mean = float(np.mean(tensor))
        tensor_shape = list(tensor.shape)
        
        preprocessing_summary = {
            "original_filename": raw_meta["filename"],
            "original_dimensions": raw_meta["dimensions"],
            "original_size": f"{raw_meta['file_size_kb']} KB",
            "target_dimensions": f"{IMG_SIZE[0]} × {IMG_SIZE[1]} px",
            "tensor_shape": tensor_shape,
            "color_channels": 3,
            "pixel_min": round(tensor_min, 4),
            "pixel_max": round(tensor_max, 4),
            "pixel_mean": round(tensor_mean, 4),
            "normalization_range": "[0.0, 1.0]",
            "validation_passed": True
        }
        
        # Step 4: Check if model weights exist
        model = load_trained_model()
        classes = get_class_labels()
        
        if model is None:
            # Model is NOT yet trained / weights not loaded
            return {
                "status": "pending_model_training",
                "is_trained": False,
                "message": (
                    "Image successfully validated, resized to 224x224, and normalized to [0.0, 1.0]! "
                    "Trained neural network weights ('model/skin_disease_model.h5') have not yet been placed into the model folder."
                ),
                "academic_note": (
                    "In adherence to B.Tech Minor Project-I academic integrity requirements, "
                    "fabricated predictions and fake accuracy numbers are strictly prohibited. "
                    "Once you train a model in Phase 2 and save weights to 'model/skin_disease_model.h5', "
                    "genuine softmax probabilities will automatically render here."
                ),
                "preprocessing_summary": preprocessing_summary,
                "target_classes": classes
            }
            
        # Step 5: Model is loaded -> Run REAL inference
        # genuine forward pass through deep neural network
        predictions = model.predict(tensor, verbose=0)
        probabilities = predictions[0]
        
        # Map predictions to class labels
        all_results = []
        for i, prob in enumerate(probabilities):
            class_info = classes[i] if i < len(classes) else {"code": f"class_{i}", "name": f"Class {i}", "type": "Unknown"}
            all_results.append({
                "code": class_info.get("code"),
                "name": class_info.get("name"),
                "type": class_info.get("type"),
                "probability": float(prob),
                "confidence_pct": round(float(prob) * 100, 2)
            })
            
        # Sort predictions by probability descending
        all_results.sort(key=lambda x: x["probability"], reverse=True)
        top_prediction = all_results[0]
        
        return {
            "status": "success",
            "is_trained": True,
            "message": "Model inference completed successfully.",
            "top_prediction": top_prediction,
            "all_predictions": all_results,
            "preprocessing_summary": preprocessing_summary
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_type": "InferenceException",
            "message": f"An unexpected error occurred during image processing: {str(e)}"
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing predictor.py module")
    # Quick self-test
    status = get_model_status()
    print("Model status from predictor:", status["message"])
    print("=" * 60)
