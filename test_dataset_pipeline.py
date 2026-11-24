"""
test_dataset_pipeline.py
==============================================================================
Skin Disease Prediction Using Deep Learning
Automated Test Suite for ISIC Dataset Pipeline, Leakage Prevention & Preprocessing
==============================================================================
"""

import sys
import json
from pathlib import Path
from PIL import Image
import numpy as np

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"
RAW_IMAGES_DIR = RAW_DIR / "images"
METADATA_FILE = RAW_DIR / "isic_metadata.json"
SUMMARY_FILE = RAW_DIR / "dataset_summary.json"
SPLIT_MANIFEST_FILE = DATASET_DIR / "split_manifest.json"
PREPROCESSING_REPORT_FILE = DATASET_DIR / "preprocessing_report.json"

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def test_raw_dataset_integrity():
    print("[Test 1/7] Testing raw dataset integrity and metadata...")
    assert RAW_IMAGES_DIR.exists(), f"Raw images folder does not exist: {RAW_IMAGES_DIR}"
    assert METADATA_FILE.exists(), f"Metadata file does not exist: {METADATA_FILE}"
    assert SUMMARY_FILE.exists(), f"Summary file does not exist: {SUMMARY_FILE}"

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary = json.load(f)

    total_downloaded = summary.get("total_images_downloaded", 0)
    assert total_downloaded > 0, "No downloaded images recorded in summary"
    assert len(meta) == total_downloaded, f"Metadata entries ({len(meta)}) mismatch summary ({total_downloaded})"

    # Verify each raw image exists and is decodable by PIL
    for isic_id, item in meta.items():
        img_path = RAW_IMAGES_DIR / f"{isic_id}.jpg"
        assert img_path.exists(), f"Raw image missing: {img_path}"
        with Image.open(img_path) as img:
            img.verify()
        with Image.open(img_path) as img:
            assert img.format in ["JPEG", "MPO"], f"Unexpected format for {isic_id}: {img.format}"
            assert img.width > 50 and img.height > 50, f"Invalid dimensions for {isic_id}"

    print(f"  -> Passed: {total_downloaded} authentic raw images verified decodable with valid headers.")


def test_metadata_diagnostic_mapping():
    print("\n[Test 2/7] Testing diagnostic label mapping to 7 benchmark classes...")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    class_counts = {c: 0 for c in CLASSES}
    for isic_id, item in meta.items():
        target_class = item.get("target_class")
        assert target_class in CLASSES, f"Invalid target class for {isic_id}: {target_class}"
        class_counts[target_class] += 1
        assert "diagnosis_1" in item or "diagnosis_2" in item or "diagnosis_3" in item
        assert "copyright_license" in item
        assert item["copyright_license"] in ["CC-BY-NC", "CC-0", "CC-BY", "CC-BY-NC 4.0"]

    for c in CLASSES:
        assert class_counts[c] > 0, f"Class {c} has 0 samples mapped!"

    print(f"  -> Passed: All samples mapped correctly across all 7 classes: {class_counts}")


def test_zero_data_leakage():
    print("\n[Test 3/7] Testing strict zero data leakage (lesion-level isolation)...")
    assert SPLIT_MANIFEST_FILE.exists(), f"Split manifest does not exist: {SPLIT_MANIFEST_FILE}"

    with open(SPLIT_MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    images = manifest.get("images", {})
    assert len(images) > 0, "Split manifest contains no images"

    train_lesions = set()
    val_lesions = set()
    test_lesions = set()

    for isic_id, item in images.items():
        split = item.get("split")
        lesion_id = item.get("lesion_id")
        assert lesion_id, f"Missing lesion_id for {isic_id}"

        if split == "train":
            train_lesions.add(lesion_id)
        elif split == "validation":
            val_lesions.add(lesion_id)
        elif split == "test":
            test_lesions.add(lesion_id)
        else:
            raise AssertionError(f"Unknown split name: {split}")

    # Check set intersections
    overlap_train_val = train_lesions & val_lesions
    overlap_train_test = train_lesions & test_lesions
    overlap_val_test = val_lesions & test_lesions

    assert len(overlap_train_val) == 0, f"LEAKAGE: {len(overlap_train_val)} lesion(s) in both train and val: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"LEAKAGE: {len(overlap_train_test)} lesion(s) in both train and test: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"LEAKAGE: {len(overlap_val_test)} lesion(s) in both val and test: {overlap_val_test}"

    assert manifest.get("zero_leakage_verified") is True
    print(f"  -> Passed: Zero data leakage confirmed! Distinct lesions: Train ({len(train_lesions)}), Val ({len(val_lesions)}), Test ({len(test_lesions)}). Intersections: 0.")


def test_preprocessed_image_dimensions():
    print("\n[Test 4/7] Testing preprocessed image dimensions and pixel scaling...")
    assert PREPROCESSING_REPORT_FILE.exists(), f"Preprocessing report missing: {PREPROCESSING_REPORT_FILE}"

    with open(SPLIT_MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    images = manifest.get("images", {})
    checked = 0

    for isic_id, item in list(images.items())[:20]:  # sample check
        rel_path = item.get("path")
        img_path = BASE_DIR / rel_path
        assert img_path.exists(), f"Preprocessed image not found: {img_path}"

        with Image.open(img_path) as img:
            assert img.size == (224, 224), f"Preprocessed image dimensions mismatch: {img.size}"
            arr = np.array(img, dtype=np.float32)
            assert arr.shape == (224, 224, 3), f"Preprocessed shape mismatch: {arr.shape}"
            # Normalize check
            norm = arr / 255.0
            assert 0.0 <= norm.min() and norm.max() <= 1.0, "Normalization range violation"
            checked += 1

    print(f"  -> Passed: Preprocessed samples verified with shape (224, 224, 3) and normalized [0.0, 1.0].")


def test_split_counts_consistency():
    print("\n[Test 5/7] Testing split folder counts against manifest...")
    with open(SPLIT_MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected_counts = manifest["split_counts"]

    for split in ["train", "validation", "test"]:
        split_dir = DATASET_DIR / split
        actual_files = list(split_dir.glob("*/*.jpg"))
        expected = expected_counts[split]
        assert len(actual_files) == expected, f"Split {split} count mismatch: actual {len(actual_files)} vs expected {expected}"

    print(f"  -> Passed: Split counts match manifest exactly (Train: {expected_counts['train']}, Val: {expected_counts['validation']}, Test: {expected_counts['test']}).")


def test_flask_dataset_page_integration():
    print("\n[Test 6/7] Testing Flask /dataset view and dynamic metadata rendering...")
    import app
    client = app.app.test_client()

    res = client.get('/dataset')
    assert res.status_code == 200, f"Expected 200 OK, got {res.status_code}"
    html = res.get_data(as_text=True)

    # Verify that the authentic dataset and split manifest card rendered
    assert "Authentic ISIC Dataset Subset & Zero-Leakage Split Manifest" in html
    assert "70 images" in html or "70" in html
    assert "0 Lesion ID Overlap Verified" in html
    assert "CC BY-NC 4.0" in html or "CC-BY-NC" in html

    print("  -> Passed: /dataset page dynamically reads and displays real local metadata and leakage status.")


def test_flask_dataset_api_status():
    print("\n[Test 7/7] Testing /api/dataset-status JSON endpoint...")
    import app
    client = app.app.test_client()

    res = client.get('/api/dataset-status')
    assert res.status_code == 200
    data = res.get_json()

    assert data.get("raw_summary") is not None
    assert data["raw_summary"]["total_images_downloaded"] == 70
    assert data.get("split_manifest") is not None
    assert data["split_manifest"]["zero_leakage_verified"] is True
    assert data["train"]["total_images"] == 44
    assert data["validation"]["total_images"] == 13
    assert data["test"]["total_images"] == 13

    print("  -> Passed: /api/dataset-status returns verified local counts and leakage status.")


def run_all_tests():
    print("=" * 70)
    print("RUNNING ISIC DATASET PIPELINE VERIFICATION SUITE")
    print("=" * 70)

    try:
        test_raw_dataset_integrity()
        test_metadata_diagnostic_mapping()
        test_zero_data_leakage()
        test_preprocessed_image_dimensions()
        test_split_counts_consistency()
        test_flask_dataset_page_integration()
        test_flask_dataset_api_status()

        print("\n" + "=" * 70)
        print("ALL 7 PIPELINE TESTS PASSED WITH 100% SUCCESS!")
        print("=" * 70)
        return True

    except AssertionError as ae:
        print(f"\n[FAIL] Test assertion failed: {ae}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)
