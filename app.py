"""
app.py
==============================================================================
Skin Disease Prediction Using Deep Learning
B.Tech Minor Project-I - Flask Web Application Entry Point
==============================================================================

Purpose:
--------
This is the main application server for the project. It provides:
1. Web routes for all pages (Home, Dataset Explorer, Methodology, Evaluation).
2. REST API endpoints for image upload, preprocessing validation, and inference.
3. Secure file upload handling with format restrictions (.jpg, .jpeg, .png).
4. Live filesystem inspection of dataset splits and model status.

Academic Integrity Guarantee:
-----------------------------
In accordance with B.Tech Minor Project-I guidelines, this server does not
produce fake predictions or fabricated accuracy scores. The data preprocessing
pipeline is live and verified, while model inference transparently indicates
when model training is pending in Phase 2.
==============================================================================
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

# Import modular application components
from model_loader import get_model_status, get_class_labels
from predictor import predict_skin_disease
from evaluation import get_evaluation_framework_info, get_reports_inventory, load_latest_evaluation_metrics
from preprocess import scan_and_validate_dataset, check_dataset_structure, DATASET_DIR

# Initialize Flask application
app = Flask(__name__)

# Base directory configuration
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
# Limit max upload size to 16 MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Allowed image extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}


def is_allowed_file(filename):
    """
    Checks if an uploaded filename has a permitted image extension.
    Permits only JPG, JPEG, and PNG.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==============================================================================
# Page Routes (HTML Views)
# ==============================================================================

@app.route('/')
def index():
    """
    Home page view:
    Displays project hero header, medical disclaimer, and the interactive
    lesion image upload, preview, and prediction card.
    """
    model_status = get_model_status()
    classes = get_class_labels()
    return render_template(
        'index.html',
        model_status=model_status,
        classes=classes,
        active_page='home'
    )


@app.route('/dataset')
def dataset():
    """
    Dataset page view:
    Presents benchmark HAM10000 metadata, real downloaded subset summary,
    reproducible split manifest, zero-leakage verification, and live filesystem audit.
    """
    # Live inspection of local dataset directory
    train_report = scan_and_validate_dataset(DATASET_DIR / "train")
    val_report = scan_and_validate_dataset(DATASET_DIR / "validation")
    test_report = scan_and_validate_dataset(DATASET_DIR / "test")
    structure = check_dataset_structure(DATASET_DIR)
    
    classes = get_class_labels()

    # Load authentic dataset metadata if present
    summary_path = DATASET_DIR / "raw" / "dataset_summary.json"
    manifest_path = DATASET_DIR / "split_manifest.json"
    report_path = DATASET_DIR / "preprocessing_report.json"

    dataset_summary = None
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                dataset_summary = json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to read {summary_path}: {e}")

    split_manifest = None
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                split_manifest = json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to read {manifest_path}: {e}")

    preprocessing_report = None
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                preprocessing_report = json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to read {report_path}: {e}")
    
    return render_template(
        'dataset.html',
        classes=classes,
        structure=structure,
        train_report=train_report,
        val_report=val_report,
        test_report=test_report,
        dataset_summary=dataset_summary,
        split_manifest=split_manifest,
        preprocessing_report=preprocessing_report,
        active_page='dataset'
    )


@app.route('/about')
def about():
    """
    Project methodology and limitations page view:
    Details the deep learning methodology, data augmentation, transfer learning,
    and vital clinical/ethical limitations (skin tone bias, dermatoscopy artifacts).
    """
    classes = get_class_labels()
    return render_template(
        'about.html',
        classes=classes,
        active_page='about'
    )


@app.route('/evaluation')
def evaluation():
    """
    Evaluation framework page view:
    Explains sensitivity, specificity, precision, F1-score, ROC-AUC,
    and displays actual test-set evaluation results when available.
    """
    framework = get_evaluation_framework_info()
    model_status = get_model_status()
    reports = get_reports_inventory()
    metrics = load_latest_evaluation_metrics()
    
    return render_template(
        'evaluation.html',
        framework=framework,
        model_status=model_status,
        reports=reports,
        metrics=metrics,
        active_page='evaluation'
    )


@app.route('/reports/<path:filename>')
def serve_report(filename):
    """Serve generated report charts and artifacts from the reports/ directory."""
    reports_dir = BASE_DIR / "reports"
    return send_from_directory(reports_dir, filename)


# ==============================================================================
# API Endpoints (JSON)
# ==============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Image upload and prediction endpoint:
    1. Validates uploaded file presence and extension (JPG, JPEG, PNG).
    2. Securely saves the image into 'uploads/'.
    3. Invokes predictor.predict_skin_disease().
    4. Returns JSON containing preprocessed tensor diagnostics and model status.
    """
    # Check if a file was included in the request
    if 'image' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file uploaded. Please select a JPG, JPEG, or PNG image."
        }), 400

    file = request.files['image']

    # Check if user submitted an empty selection
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No file selected. Please select a valid skin lesion image."
        }), 400

    # Validate file extension
    if not is_allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "message": "Unsupported file format. Please upload a JPG, JPEG, or PNG image."
        }), 400

    try:
        # Save file securely
        filename = secure_filename(file.filename)
        # Avoid empty filename from odd unicode names
        if not filename:
            filename = "uploaded_skin_lesion.jpg"
            
        saved_filepath = UPLOAD_FOLDER / filename
        file.save(str(saved_filepath))

        # Execute prediction and preprocessing pipeline
        result = predict_skin_disease(saved_filepath)
        
        # Include relative path for previewing in response if needed
        result["uploaded_filename"] = filename
        
        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Server error processing image: {str(e)}"
        }), 500


@app.route('/api/dataset-status', methods=['GET'])
def api_dataset_status():
    """
    Returns live statistics on local images in dataset/train, dataset/val, dataset/test,
    as well as raw collection summary and split manifest status.
    """
    structure = check_dataset_structure(DATASET_DIR)
    train_report = scan_and_validate_dataset(DATASET_DIR / "train")
    val_report = scan_and_validate_dataset(DATASET_DIR / "validation")
    test_report = scan_and_validate_dataset(DATASET_DIR / "test")

    summary_path = DATASET_DIR / "raw" / "dataset_summary.json"
    manifest_path = DATASET_DIR / "split_manifest.json"

    raw_summary = None
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                raw_summary = json.load(f)
        except Exception:
            pass

    manifest_info = None
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)
                manifest_info = {
                    "total_images": m.get("total_images"),
                    "split_counts": m.get("split_counts"),
                    "zero_leakage_verified": m.get("zero_leakage_verified"),
                    "leakage_verification_status": m.get("leakage_verification_status"),
                    "limitations_noted": m.get("limitations_noted")
                }
        except Exception:
            pass
    
    return jsonify({
        "structure": structure,
        "train": train_report,
        "validation": val_report,
        "test": test_report,
        "raw_summary": raw_summary,
        "split_manifest": manifest_info
    })


@app.route('/api/model-status', methods=['GET'])
def api_model_status():
    """
    Returns live model weights loading status.
    """
    return jsonify(get_model_status())


# ==============================================================================
# Main Entry Point
# ==============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Starting Flask Web Application for Skin Disease Prediction")
    print("B.Tech Minor Project-I")
    print(f"Server accessible at: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(debug=True, host='127.0.0.1', port=5000)
