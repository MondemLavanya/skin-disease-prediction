"""
preprocess.py
==============================================================================
Skin Disease Prediction Using Deep Learning
B.Tech Minor Project - Preprocessing & Dataset Management Module
==============================================================================

This module handles:
1. Dataset folder verification and structure validation
2. Class-name detection from folder hierarchy
3. Image integrity verification (identifies corrupted/unreadable files)
4. Image resizing to standard input shape (224 x 224)
5. Pixel normalization (scaling pixel values to [0.0, 1.0])
6. Data augmentation pipeline for training (rotation, flip, zoom, contrast, brightness)
7. Training, validation, and test dataset loaders using tf.data and Keras
8. Single image loading and preprocessing for inference

Target Directory Layout:
dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── validation/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
==============================================================================
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# ==============================================================================
# Configuration Constants
# ==============================================================================
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
BATCH_SIZE = 32

# Common supported image extensions in medical/computer vision datasets
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Default project directories
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"


# ==============================================================================
# 1. Dataset Folder & Class Name Handling
# ==============================================================================
def get_class_names(directory_path):
    """
    Detects and returns sorted class names based on subdirectories in the given path.
    
    Parameters:
        directory_path (str or Path): Path to dataset split (e.g., dataset/train).
        
    Returns:
        list[str]: Sorted list of subfolder names representing class labels.
    """
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        print(f"[Warning] Directory does not exist: {directory}")
        return []
    
    # Subdirectories that are not hidden/special (excluding .gitkeep or other hidden files)
    classes = sorted([d.name for d in directory.iterdir() if d.is_dir() and not d.name.startswith('.')])
    return classes


def check_dataset_structure(dataset_dir=DATASET_DIR):
    """
    Verifies that the required train, validation, and test folders exist.
    Checks class consistency across splits.
    
    Parameters:
        dataset_dir (str or Path): Root path to dataset folder.
        
    Returns:
        dict: Summary of dataset status including found splits and class names.
    """
    dataset_path = Path(dataset_dir)
    splits = {
        "train": dataset_path / "train",
        "validation": dataset_path / "validation",
        "test": dataset_path / "test"
    }
    
    status = {
        "dataset_exists": dataset_path.exists(),
        "splits_exist": {},
        "classes": {}
    }
    
    for split_name, split_path in splits.items():
        exists = split_path.exists() and split_path.is_dir()
        status["splits_exist"][split_name] = exists
        if exists:
            status["classes"][split_name] = get_class_names(split_path)
        else:
            status["classes"][split_name] = []
            
    return status


# ==============================================================================
# 2. Image Validation & Integrity Checking
# ==============================================================================
def validate_image_file(image_path):
    """
    Checks if a single file exists, has a valid image extension,
    and can be successfully opened and decoded by PIL without corruption.
    
    Parameters:
        image_path (str or Path): Path to the image file.
        
    Returns:
        tuple (bool, str or None): (True, None) if valid, (False, error_reason) if invalid.
    """
    path = Path(image_path)
    if not path.is_file():
        return False, "File not found or not a regular file."
    
    if path.suffix.lower() not in VALID_EXTENSIONS:
        return False, f"Unsupported file extension: {path.suffix}"
    
    try:
        # Step 1: Open and verify file integrity
        with Image.open(path) as img:
            img.verify()
        
        # Step 2: Open and load pixel data (detects truncated or partial files)
        with Image.open(path) as img:
            img.load()
            # Ensure it can be converted to RGB format (handles CMYK, RGBA, Grayscale)
            _ = img.convert('RGB')
            
        return True, None
    except Exception as e:
        return False, f"Corrupted or unreadable image: {str(e)}"


def scan_and_validate_dataset(directory_path):
    """
    Scans a dataset directory (like train/ or test/), validates every image file,
    and returns a summary of class distribution and any invalid/corrupted files.
    
    Parameters:
        directory_path (str or Path): Directory to scan.
        
    Returns:
        dict: Inspection report containing image counts per class and list of corrupted files.
    """
    directory = Path(directory_path)
    report = {
        "directory": str(directory),
        "total_images": 0,
        "class_counts": {},
        "corrupted_files": [],
        "unsupported_files": []
    }
    
    if not directory.exists():
        print(f"[Warning] Directory not found: {directory}")
        return report

    classes = get_class_names(directory)
    for class_name in classes:
        class_folder = directory / class_name
        valid_count = 0
        for item in class_folder.iterdir():
            if item.is_file() and not item.name.startswith('.'):
                is_valid, error = validate_image_file(item)
                if is_valid:
                    valid_count += 1
                else:
                    if "extension" in (error or "").lower():
                        report["unsupported_files"].append((str(item), error))
                    else:
                        report["corrupted_files"].append((str(item), error))
        report["class_counts"][class_name] = valid_count
        report["total_images"] += valid_count
        
    return report


# ==============================================================================
# 3. Augmentation & Normalization Pipelines
# ==============================================================================
def create_data_augmentation_pipeline():
    """
    Builds a TensorFlow/Keras Sequential model for data augmentation.
    
    Augmentations applied:
    - RandomFlip("horizontal"): Flips images horizontally (useful for symmetric dermatological patterns)
    - RandomRotation(0.1): Randomly rotates images by up to +/- 10% (36 degrees)
    - RandomZoom(0.1): Randomly zooms in/out by up to 10%
    - RandomContrast(0.1): Randomly adjusts contrast by up to 10%
    - RandomBrightness(0.1): Small, safe brightness variation (if supported in TF/Keras)
    
    Returns:
        tf.keras.Sequential: Keras preprocessing layers wrapped in a sequential model.
    """
    augmentation_layers = [
        layers.RandomFlip("horizontal", name="aug_horizontal_flip"),
        layers.RandomRotation(0.1, fill_mode="reflect", name="aug_rotation"),
        layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="reflect", name="aug_zoom"),
        layers.RandomContrast(0.1, name="aug_contrast")
    ]
    
    # Safely add RandomBrightness if available in this Keras version
    if hasattr(layers, "RandomBrightness"):
        try:
            augmentation_layers.append(layers.RandomBrightness(factor=0.1, value_range=(0.0, 255.0), name="aug_brightness"))
        except Exception:
            # Value range might vary depending on backend/version; skip safely if initialization fails
            pass
            
    return tf.keras.Sequential(augmentation_layers, name="data_augmentation_pipeline")


def create_rescaling_layer():
    """
    Creates a Keras Rescaling layer that maps pixel values from [0, 255] to [0.0, 1.0].
    
    Returns:
        tf.keras.layers.Rescaling
    """
    return layers.Rescaling(scale=1.0 / 255.0, name="pixel_rescaling")


# ==============================================================================
# 4. TensorFlow Dataset Loading (Train / Validation / Test)
# ==============================================================================
def load_dataset_from_directory(directory_path, 
                                 image_size=IMG_SIZE, 
                                 batch_size=BATCH_SIZE, 
                                 shuffle=True, 
                                 label_mode='categorical'):
    """
    Loads an image dataset from a directory using tf.keras.utils.image_dataset_from_directory.
    
    Parameters:
        directory_path (str or Path): Path to split directory containing subfolders per class.
        image_size (tuple): Target (height, width) to resize images to (default: (224, 224)).
        batch_size (int): Number of images per batch (default: 32).
        shuffle (bool): Whether to shuffle the images (True for training, False for test/val).
        label_mode (str): 'categorical' for one-hot encoded labels, or 'sparse' for integer labels.
        
    Returns:
        tuple (tf.data.Dataset or None, list[str]): The loaded dataset and the list of detected class names.
    """
    path = Path(directory_path)
    if not path.exists():
        print(f"[Error] Dataset directory does not exist: {path}")
        return None, []
    
    classes = get_class_names(path)
    if not classes:
        print(f"[Warning] No class subfolders found inside: {path}")
        return None, []
    
    try:
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory=str(path),
            labels='inferred',
            label_mode=label_mode,
            class_names=classes,
            color_mode='rgb',
            batch_size=batch_size,
            image_size=image_size,
            shuffle=shuffle,
            interpolation='bilinear'
        )
        return dataset, classes
    except ValueError as ve:
        # Happens if folders exist but contain no valid images yet
        print(f"[Warning] Could not load dataset from {path}: {ve}")
        return None, classes


def prepare_dataset_pipeline(dataset, 
                             is_training=False, 
                             apply_augmentation=True, 
                             apply_rescaling=True):
    """
    Optimizes a tf.data.Dataset pipeline:
    - Applies normalization (rescaling pixel values to [0, 1])
    - Applies data augmentation (only if is_training=True and apply_augmentation=True)
    - Caches dataset in memory for speed
    - Prefetches batches using tf.data.AUTOTUNE to prevent GPU/CPU bottlenecks
    
    Parameters:
        dataset (tf.data.Dataset): The raw dataset loaded from disk.
        is_training (bool): Flag indicating if this is the training split.
        apply_augmentation (bool): Whether to apply augmentation layers.
        apply_rescaling (bool): Whether to apply [0, 1] normalization.
        
    Returns:
        tf.data.Dataset: The optimized dataset.
    """
    if dataset is None:
        return None
    
    rescale_layer = create_rescaling_layer()
    aug_pipeline = create_data_augmentation_pipeline() if (is_training and apply_augmentation) else None
    
    # 1. Normalization & optional Augmentation mapping
    def transform(images, labels):
        if is_training and aug_pipeline is not None:
            images = aug_pipeline(images, training=True)
        if apply_rescaling:
            images = rescale_layer(images)
        return images, labels
    
    dataset = dataset.map(transform, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 2. Performance optimization: Cache and Prefetch
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset


def load_all_datasets(dataset_dir=DATASET_DIR, 
                      image_size=IMG_SIZE, 
                      batch_size=BATCH_SIZE, 
                      apply_augmentation=True):
    """
    High-level convenience function to load all splits (train, validation, test),
    apply preprocessing/augmentation, and return datasets ready for model training.
    
    Returns:
        dict: {
            "train_ds": tf.data.Dataset or None,
            "val_ds": tf.data.Dataset or None,
            "test_ds": tf.data.Dataset or None,
            "class_names": list[str]
        }
    """
    dataset_path = Path(dataset_dir)
    train_path = dataset_path / "train"
    val_path = dataset_path / "validation"
    test_path = dataset_path / "test"
    
    # Load raw datasets
    raw_train, train_classes = load_dataset_from_directory(
        train_path, image_size=image_size, batch_size=batch_size, shuffle=True
    )
    raw_val, _ = load_dataset_from_directory(
        val_path, image_size=image_size, batch_size=batch_size, shuffle=False
    )
    raw_test, _ = load_dataset_from_directory(
        test_path, image_size=image_size, batch_size=batch_size, shuffle=False
    )
    
    # Prepare optimized pipelines
    train_ds = prepare_dataset_pipeline(raw_train, is_training=True, apply_augmentation=apply_augmentation)
    val_ds = prepare_dataset_pipeline(raw_val, is_training=False, apply_augmentation=False)
    test_ds = prepare_dataset_pipeline(raw_test, is_training=False, apply_augmentation=False)
    
    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "class_names": train_classes
    }


# ==============================================================================
# 5. Single Image Preprocessing (For Inference / Predict Module)
# ==============================================================================
def load_and_preprocess_single_image(image_path, target_size=IMG_SIZE, normalize=True):
    """
    Loads, validates, resizes, and normalizes a single image file for model prediction.
    
    Parameters:
        image_path (str or Path): Path to the image file to be predicted.
        target_size (tuple): Target (height, width) dimensions, default (224, 224).
        normalize (bool): If True, divides pixel values by 255.0 to scale to [0, 1].
        
    Returns:
        np.ndarray: Preprocessed image tensor with shape (1, 224, 224, 3) ready for model.predict().
        
    Raises:
        ValueError: If the file is invalid, corrupted, or unsupported.
    """
    is_valid, error = validate_image_file(image_path)
    if not is_valid:
        raise ValueError(f"Cannot preprocess invalid image '{image_path}': {error}")
    
    # Load image and ensure RGB 3-channel mode
    img = Image.open(image_path).convert('RGB')
    
    # Resize using high-quality Lanczos resampling
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert PIL Image to float32 NumPy array
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize pixel values from [0, 255] to [0.0, 1.0]
    if normalize:
        img_array = img_array / 255.0
        
    # Expand dimensions to create batch shape: (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch
