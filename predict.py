"""
predict.py
======================================================
Skin Disease Prediction Using Deep Learning
B.Tech Minor Project

Module for:
1. Loading the trained deep learning model
2. Image preprocessing (resizing, normalization via preprocess.py)
3. Running model inference
4. Mapping output probabilities to disease class labels
======================================================
"""

import os
from preprocess import (
    load_and_preprocess_single_image,
    validate_image_file,
    IMG_SIZE
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'skin_disease_model.h5')

# Define placeholder class labels (to be populated based on dataset/training)
CLASS_LABELS = []


def load_prediction_model(model_path=MODEL_PATH):
    """Load the trained Keras/TensorFlow model."""
    # TODO: Implement tf.keras.models.load_model in the prediction phase
    pass


def preprocess_image(image_path, target_size=IMG_SIZE):
    """
    Validate, resize to (224, 224), and normalize a single input image.
    Uses load_and_preprocess_single_image from preprocess.py.
    """
    return load_and_preprocess_single_image(image_path, target_size=target_size, normalize=True)


def predict_disease(image_path):
    """
    Run model inference on a single image and return predicted class and confidence.
    """
    # Validate image first
    is_valid, error = validate_image_file(image_path)
    if not is_valid:
        return {"error": f"Invalid image file: {error}"}
    
    # Preprocess image
    image_tensor = preprocess_image(image_path)
    print(f"Preprocessed image ready for model inference. Shape: {image_tensor.shape}")
    
    # Placeholder prediction response until model weights exist
    return {
        "disease": "Placeholder Disease (Pending Model Training)",
        "confidence": 0.0
    }


if __name__ == '__main__':
    print("Prediction module with integrated preprocessing ready.")
