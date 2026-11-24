"""
train_model.py
==============================================================================
Skin Disease Prediction Using Deep Learning
B.Tech Minor Project-I - Deep Learning Model Training Pipeline (Phase 3A)
==============================================================================

Architecture Recommendation: MobileNetV2 (Transfer Learning)
-------------------------------------------------------------
Why MobileNetV2?
1. Pre-trained on ImageNet (1.4M images, 1,000 general object classes) providing
   hierarchical edge, texture, and visual pattern representations.
2. Lightweight and efficient: ~2.2M total parameters (~330K trainable in Phase 1),
   making it ideal for academic projects and rapid convergence without massive compute clusters.
3. Inverted Residuals & Linear Bottlenecks preserve fine-grained dermatoscopic feature maps.
4. Input resolution: (224, 224, 3) matching standardized dermoscopy preprocessing.

Class-Imbalance Strategy:
-------------------------
In HAM10000, Melanocytic Nevi (nv) represents 67% of samples, while Melanoma (mel) is 11%,
and Dermatofibroma (df) is ~1%. Unweighted cross-entropy causes models to bias heavily toward 'nv'.
We compute balanced class weights using scikit-learn:
    weight_c = N / (n_classes * N_c)
This heavily penalizes misclassification of malignant lesions (mel, bcc) and rare classes.

Academic Integrity & Safety Enforcement:
----------------------------------------
Per Minor Project-I requirements:
- Training is strictly guarded. It does NOT train on the 70-image development subset
  or fabricate high accuracy numbers from small samples.
- Requires either the full dataset or explicit developer confirmation.
==============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight

# Import preprocessing utilities
from preprocess import (
    load_all_datasets,
    load_dataset_from_directory,
    prepare_dataset_pipeline,
    check_dataset_structure,
    get_class_names,
    IMG_SIZE,
    IMG_SHAPE,
    BATCH_SIZE,
    DATASET_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR
)

# Project Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_SAVE_PATH = MODEL_DIR / "skin_disease_model.h5"
MODEL_INFO_FILE = MODEL_DIR / "model_info.json"
REPORTS_DIR = BASE_DIR / "reports"

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASSES)


def build_mobilenet_v2_model(num_classes=NUM_CLASSES, 
                             input_shape=(224, 224, 3), 
                             dropout_rate=0.30, 
                             fine_tune_at=None):
    """
    Constructs the Transfer Learning model using MobileNetV2 pre-trained on ImageNet.
    
    Architecture:
    1. Input layer: (224, 224, 3)
    2. Data Augmentation: RandomFlip, RandomRotation(0.1), RandomZoom(0.1), RandomContrast(0.1)
    3. MobileNetV2 Backbone (frozen initially)
    4. GlobalAveragePooling2D
    5. BatchNormalization
    6. Dense(256, activation='relu')
    7. Dropout(0.30)
    8. Dense(7, activation='softmax', name='disease_prediction')
    
    Parameters:
        num_classes (int): Number of diagnostic categories (7 for HAM10000).
        input_shape (tuple): Spatial dimensions and color channels.
        dropout_rate (float): Regularization rate.
        fine_tune_at (int or None): Layer index from which to unfreeze for fine-tuning.
        
    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    print(f"[Model Architecture] Building MobileNetV2 backbone (Input shape: {input_shape})...")

    # 1. Base Pre-trained Model
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # 2. Freeze base layers for initial feature extraction
    if fine_tune_at is None:
        base_model.trainable = False
        print("      -> Base MobileNetV2 layers FROZEN for initial transfer learning.")
    else:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        print(f"      -> Unfreezing MobileNetV2 layers from layer {fine_tune_at} onwards for fine-tuning.")

    # 3. Input & Augmentation layers
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    
    # Optional Data Augmentation embedded in graph
    augmented = layers.RandomFlip("horizontal")(inputs)
    augmented = layers.RandomRotation(0.1, fill_mode="reflect")(augmented)
    augmented = layers.RandomZoom((-0.1, 0.1), fill_mode="reflect")(augmented)
    augmented = layers.RandomContrast(0.1)(augmented)

    # 4. Feature Extraction
    features = base_model(augmented, training=False)
    pooled = layers.GlobalAveragePooling2D(name="global_avg_pool")(features)
    normed = layers.BatchNormalization(name="batch_norm")(pooled)
    
    # 5. Classification Head
    dense_1 = layers.Dense(256, activation="relu", name="dense_256")(normed)
    dropped = layers.Dropout(dropout_rate, name="dropout_head")(dense_1)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_prediction")(dropped)

    # 6. Assemble Model
    model = models.Model(inputs=inputs, outputs=outputs, name="SkinDisease_MobileNetV2")

    # 7. Compile with Adam and Cross-Entropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.CategoricalCrossentropy(name="cat_loss"),
            tf.keras.metrics.AUC(name="auc", multi_label=True)
        ]
    )

    trainable_count = sum(p.numpy().size for p in model.trainable_variables)
    non_trainable_count = sum(p.numpy().size for p in model.non_trainable_variables)
    print(f"      -> Total Parameters: {trainable_count + non_trainable_count:,}")
    print(f"      -> Trainable Parameters: {trainable_count:,}")
    print(f"      -> Non-trainable Parameters: {non_trainable_count:,}")
    return model


def calculate_class_weights_from_manifest(manifest_path=DATASET_DIR / "split_manifest.json"):
    """
    Computes balanced class weights to counteract the severe class imbalance in HAM10000:
    weight = total_samples / (n_classes * class_samples)
    
    Returns:
        dict: Mapping of class index (0 to 6) to numerical weight.
    """
    print("[Class Imbalance] Computing balanced class weights from split manifest...")
    if not manifest_path.exists():
        print(f"[Warning] Manifest {manifest_path} not found. Returning default equal weights.")
        return {i: 1.0 for i in range(NUM_CLASSES)}

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Extract training samples
    train_images = [item for item in manifest.get("images", {}).values() if item.get("split") == "train"]
    if not train_images:
        return {i: 1.0 for i in range(NUM_CLASSES)}

    y_train = [CLASSES.index(item["class"]) for item in train_images if item["class"] in CLASSES]
    unique_classes = np.unique(y_train)

    computed = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y_train
    )

    weights_dict = {int(cls_idx): float(computed[i]) for i, cls_idx in enumerate(unique_classes)}
    print("      -> Calculated Class Weights:")
    for idx, w in sorted(weights_dict.items()):
        print(f"         Class {idx} ({CLASSES[idx]}): {w:.3f}")
    return weights_dict


def get_training_callbacks(save_path=MODEL_SAVE_PATH, patience=6):
    """
    Sets up training callbacks:
    1. EarlyStopping: halts training when validation loss stops improving to prevent overfitting.
    2. ModelCheckpoint: saves strictly the best weights based on validation loss.
    3. ReduceLROnPlateau: decreases learning rate by factor of 0.2 if plateau occurs.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=str(save_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]


def inspect_training_prerequisites():
    """
    Checks dataset status, image counts, hardware capability, and confirms readiness.
    """
    manifest_path = DATASET_DIR / "split_manifest.json"
    total_train = len(list(TRAIN_DIR.glob("*/*.jpg")))
    total_val = len(list(VAL_DIR.glob("*/*.jpg")))
    total_test = len(list(TEST_DIR.glob("*/*.jpg")))
    total_images = total_train + total_val + total_test

    status = {
        "total_images": total_images,
        "train_count": total_train,
        "val_count": total_val,
        "test_count": total_test,
        "is_dev_subset": total_images < 500,
        "full_dataset_present": total_images >= 9000
    }
    return status


def main():
    parser = argparse.ArgumentParser(description="Skin Disease Deep Learning Model Trainer (MobileNetV2)")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Inspect model architecture, parameters, and class weights without fitting")
    parser.add_argument("--force-train-dev-subset", action="store_true", default=False,
                        help="Override safety guard to train on small development subset (NOT recommended for project evaluation)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    args = parser.parse_args()

    print("=" * 70)
    print("SKIN DISEASE PREDICTION: MOBILENETV2 TRAINING PIPELINE (PHASE 3A)")
    print("=" * 70)

    # 1. Inspect prerequisites
    prereqs = inspect_training_prerequisites()
    print(f"Current local dataset images: {prereqs['total_images']} "
          f"(Train: {prereqs['train_count']}, Val: {prereqs['val_count']}, Test: {prereqs['test_count']})")

    # 2. Build and verify MobileNetV2 architecture
    model = build_mobilenet_v2_model(num_classes=NUM_CLASSES)
    
    # 3. Compute class weights
    class_weights = calculate_class_weights_from_manifest()

    # 4. Safety Guard Check
    if prereqs["is_dev_subset"] and not args.force_train_dev_subset:
        print("\n" + "=" * 70)
        print("ACADEMIC INTEGRITY SAFETY GUARD ACTIVATED")
        print("=" * 70)
        print(f"Local dataset contains only {prereqs['total_images']} images (Development Testing Subset).")
        print("Per B.Tech Minor Project-I guidelines:")
        print("  - Training on a 70-image sample will cause severe overfitting.")
        print("  - Reporting accuracy or sensitivity numbers from a 70-image model is statistically invalid.")
        print("  - To train a clinically reliable model, please train on the complete 10,015 HAM10000 dataset")
        print("    using Google Colab GPU (see train_on_colab.ipynb or instructions below).")
        print("\nModel architecture is compiled and ready for Phase 3 training once full data is present.")
        print("Dry run completed successfully.")
        return

    if not args.dry_run and (prereqs["full_dataset_present"] or args.force_train_dev_subset):
        print(f"\n[Training] Starting MobileNetV2 training for {args.epochs} epochs...")
        # Training code execution (deferred to Phase 3B)


if __name__ == "__main__":
    main()
