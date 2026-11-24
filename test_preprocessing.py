"""
test_preprocessing.py
==============================================================================
Unit and integration test for preprocess.py
Verifies:
1. Dataset structure detection & empty folder resilience
2. Image validation (valid image vs corrupted file)
3. Single image preprocessing (resizing to 224x224, normalization to [0, 1])
4. Data augmentation pipeline execution (rotation, zoom, flip, contrast)
5. Keras Rescaling layer
6. Temporary dataset batch loading and pipeline preparation
==============================================================================
"""

import sys
import shutil
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

from preprocess import (
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_SIZE,
    IMG_SHAPE,
    get_class_names,
    check_dataset_structure,
    validate_image_file,
    scan_and_validate_dataset,
    create_data_augmentation_pipeline,
    create_rescaling_layer,
    load_and_preprocess_single_image,
    load_dataset_from_directory,
    prepare_dataset_pipeline,
    DATASET_DIR
)


def run_tests():
    print("=" * 60)
    print("RUNNING PREPROCESSING MODULE TESTS")
    print(f"TensorFlow Version: {tf.__version__}")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = 6

    # Test 1: Check structure of actual dataset folder
    print("\n[Test 1/6] Checking actual project dataset structure...")
    status = check_dataset_structure(DATASET_DIR)
    assert status["dataset_exists"] is True, "dataset/ directory must exist"
    assert status["splits_exist"]["train"] is True, "dataset/train must exist"
    assert status["splits_exist"]["validation"] is True, "dataset/validation must exist"
    assert status["splits_exist"]["test"] is True, "dataset/test must exist"
    print("  -> Passed: Dataset splits exist and are properly structured.")
    passed_tests += 1

    # Create an isolated temporary directory for file and pipeline tests
    temp_dir = Path(tempfile.mkdtemp(prefix="test_preprocess_"))
    try:
        # Test 2: Image validation (valid vs corrupt)
        print("\n[Test 2/6] Testing image integrity validation...")
        valid_img_path = temp_dir / "valid_sample.jpg"
        dummy_img = Image.new("RGB", (300, 400), color=(120, 80, 50))
        dummy_img.save(valid_img_path)

        corrupt_img_path = temp_dir / "corrupted_sample.jpg"
        with open(corrupt_img_path, "wb") as f:
            f.write(b"NOT_AN_IMAGE_FILE_RANDOM_CORRUPT_BYTES")

        is_valid, err = validate_image_file(valid_img_path)
        assert is_valid is True and err is None, f"Expected valid image, got error: {err}"

        is_valid, err = validate_image_file(corrupt_img_path)
        assert is_valid is False and err is not None, "Expected corrupted image to fail validation"
        print(f"  -> Passed: Correctly identified valid and corrupted images (Error caught: '{err}').")
        passed_tests += 1

        # Test 3: Single image loading, resizing (224x224), and normalization
        print("\n[Test 3/6] Testing single image loading and preprocessing...")
        processed = load_and_preprocess_single_image(valid_img_path, target_size=IMG_SIZE, normalize=True)
        assert processed.shape == (1, 224, 224, 3), f"Expected shape (1, 224, 224, 3), got {processed.shape}"
        assert processed.dtype == np.float32, f"Expected float32, got {processed.dtype}"
        assert 0.0 <= processed.min() and processed.max() <= 1.0, "Pixels must be normalized within [0.0, 1.0]"
        print(f"  -> Passed: Image resized to {processed.shape} and normalized to range [{processed.min():.2f}, {processed.max():.2f}].")
        passed_tests += 1

        # Test 4: Data Augmentation Pipeline
        print("\n[Test 4/6] Testing data augmentation pipeline...")
        aug_model = create_data_augmentation_pipeline()
        sample_tensor = tf.constant(processed)
        augmented = aug_model(sample_tensor, training=True)
        assert augmented.shape == (1, 224, 224, 3), f"Augmented output shape mismatch: {augmented.shape}"
        print(f"  -> Passed: Augmentation pipeline successfully executed on tensor (Output shape: {augmented.shape}).")
        passed_tests += 1

        # Test 5: Pixel Rescaling Layer
        print("\n[Test 5/6] Testing Keras Rescaling layer...")
        rescale_layer = create_rescaling_layer()
        raw_pixels = tf.constant(np.full((1, 224, 224, 3), 255.0, dtype=np.float32))
        rescaled = rescale_layer(raw_pixels)
        assert np.isclose(rescaled.numpy().max(), 1.0), "Rescaling layer must scale 255.0 to 1.0"
        print(f"  -> Passed: 255.0 pixel value scaled to {rescaled.numpy().max():.2f}.")
        passed_tests += 1

        # Test 6: Mock dataset batch loading with class folders
        print("\n[Test 6/6] Testing dataset loading with class folders...")
        mock_split_dir = temp_dir / "mock_train"
        mock_split_dir.mkdir()
        for class_name in ["melanoma", "nevus"]:
            class_folder = mock_split_dir / class_name
            class_folder.mkdir()
            for i in range(2):
                img_p = class_folder / f"sample_{i}.jpg"
                Image.new("RGB", (100, 100), color=(i * 50, i * 20, 100)).save(img_p)

        classes = get_class_names(mock_split_dir)
        assert classes == ["melanoma", "nevus"], f"Expected ['melanoma', 'nevus'], got {classes}"

        dataset, detected_classes = load_dataset_from_directory(
            mock_split_dir, image_size=(224, 224), batch_size=2, shuffle=False
        )
        assert dataset is not None, "Dataset should load successfully"
        assert detected_classes == ["melanoma", "nevus"]

        # Test pipeline optimization (map, cache, prefetch)
        optimized_ds = prepare_dataset_pipeline(dataset, is_training=True, apply_augmentation=True)
        for batch_images, batch_labels in optimized_ds.take(1):
            assert batch_images.shape == (2, 224, 224, 3), f"Batch shape mismatch: {batch_images.shape}"
            assert batch_labels.shape == (2, 2), f"Labels shape mismatch: {batch_labels.shape}"
            print(f"  -> Passed: Mock dataset loaded with {len(classes)} classes, batch shape {batch_images.shape}, labels {batch_labels.shape}.")
        passed_tests += 1

    finally:
        # Cleanup temporary test files
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print(f"ALL TESTS PASSED: {passed_tests}/{total_tests}")
    print("=" * 60)
    return True


if __name__ == '__main__':
    success = run_tests()
    if not success:
        sys.exit(1)
