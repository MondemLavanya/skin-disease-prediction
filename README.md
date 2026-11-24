# Skin Disease Prediction Using Deep Learning

**B.Tech Minor Project-I | Computer Science & Engineering (Artificial Intelligence & Machine Learning)**

---

## 📌 Project Overview

This project is a healthcare-focused computer vision web application developed for **Minor Project-I**. The primary goal is to provide an automated, assistive decision-support interface capable of ingesting dermatoscopic skin lesion images, validating and preprocessing them into standardized normalized tensors, and preparing for deep convolutional neural network (CNN) inference against the benchmark **HAM10000** dataset.

### Academic Integrity & Safety Notice:
> **No Fabricated Predictions**: In adherence to strict academic and medical ethics standards, this application **does not produce fake diagnostic predictions or hardcode bogus accuracy scores**.
> - The image validation and preprocessing pipeline (`(224, 224, 3)`, normalized to `[0.0, 1.0]`) is **100% active and verified**.
> - When model weights (`model/skin_disease_model.h5`) are pending from Phase 2 training, the system honestly reports pre-training verification diagnostics.
> - Once weights are saved into `model/`, the inference engine automatically detects them and outputs genuine softmax probabilities.

---

## 📂 Project Directory Structure & Purpose of Major Files

```text
skin-disease-prediction/
│
├── dataset/                     # Storage for dermatoscopic datasets
│   ├── raw/                     # Original untouched files & metadata from ISIC
│   │   ├── images/              # Authentic downloaded raw JPG files (70 development images)
│   │   ├── isic_metadata.json   # Complete clinical metadata per image
│   │   └── dataset_summary.json # Summary: source, license, total counts, class distribution
│   │
│   ├── train/                   # Leakage-free training split (44 images across 7 classes)
│   ├── validation/              # Leakage-free validation split (13 images across 7 classes)
│   ├── test/                    # Leakage-free test split (13 images across 7 classes)
│   │
│   ├── split_manifest.json      # Reproducible mapping: image_id -> lesion_id -> split -> class
│   └── preprocessing_report.json# Preprocessing report: shape, resolution, normalization
│
├── model/                       # Model storage and architecture metadata
│   ├── .gitkeep                 # Folder placeholder
│   └── model_info.json          # Metadata defining the 7 HAM10000 classes & input shape
│
├── reports/                     # Output directory for evaluation reports and charts
│   └── .gitkeep
│
├── static/                      # Static assets served by Flask
│   ├── css/
│   │   └── style.css            # Responsive, healthcare-themed CSS (clinical teal palette)
│   └── js/
│       └── main.js              # Drag-and-drop, client-side preview, AJAX inference
│
├── templates/                   # Jinja2 HTML templates for the web interface
│   ├── base.html                # Base layout: navbar, clinical disclaimer, footer
│   ├── index.html               # Home page: lesion upload, preview, and results panel
│   ├── dataset.html             # HAM10000 benchmark overview, raw summary, and split manifest
│   ├── about.html               # Project methodology, CNN pipeline, and clinical limitations
│   └── evaluation.html          # Medical evaluation metrics guide (Sensitivity vs Accuracy)
│
├── uploads/                     # Temporary folder for user-uploaded query images
│
├── download_dataset.py          # Script to download authentic images & metadata from ISIC API v2
├── dataset_preprocessor.py      # Lesion-level leakage-free splitter & Lanczos 224x224 preprocessor
├── test_dataset_pipeline.py     # Automated test suite for data integrity & zero leakage
├── test_app.py                  # Integration test suite for Flask routes and predict endpoint
├── test_preprocessing.py        # Unit tests for preprocessing & data augmentation
├── preprocess.py                # Single image preprocessing & tf.data batch loaders
├── model_loader.py              # Safe model discovery, in-memory caching, metadata loader
├── predictor.py                 # Prediction orchestrator (preprocessing + inference flow)
├── evaluation.py                # Clinical metrics framework (Sensitivity, Specificity, F1)
├── app.py                       # Flask server entry point & REST API routes
├── requirements.txt             # Python dependencies
└── README.md                    # Setup guide and project documentation
```

### Purpose of Every Major File:
1. **`download_dataset.py`**: Connects to the official ISIC Archive API v2 (HAM10000 Collection #212) and downloads authentic dermatoscopic images directly from authorized AWS S3 storage under the CC BY-NC 4.0 license, saving untouched originals in `dataset/raw/images/`.
2. **`dataset_preprocessor.py`**: Performs clinical diagnostic label mapping, partitions samples into Train/Val/Test by `lesion_id` to guarantee zero data leakage, resizes to `(224, 224)` via Lanczos resampling, and saves `dataset/split_manifest.json`.
3. **`test_dataset_pipeline.py`**: Automated test suite that validates raw image decodability, lesion-level split isolation (0 overlap), preprocessed array shapes, and website synchronization.
4. **`app.py`**: The central Flask application. Manages page routes (`/`, `/dataset`, `/about`, `/evaluation`), enforces secure file uploads (`.jpg`, `.jpeg`, `.png`), and exposes the `/predict` API endpoint.
5. **`preprocess.py`**: The image engineering module. Validates file integrity using PIL, resizes images to `224 × 224` pixels using Lanczos filtering, rescales pixel intensities from `[0, 255]` to `[0.0, 1.0]`, and defines data augmentation layers (flips, rotations, zoom).
6. **`model_loader.py`**: Safely searches for and loads trained Keras weights (`.h5` or `.keras`). Caches the loaded model in RAM so predictions run quickly without reloading from disk every time.
7. **`predictor.py`**: Connects preprocessing with model loading. Produces verified tensor diagnostics and handles both pre-training and trained states gracefully.
8. **`evaluation.py`**: Defines medical evaluation metrics (Sensitivity, Specificity, Precision, F1-Score, ROC-AUC) and explains why raw accuracy is misleading on imbalanced clinical datasets.
9. **`templates/`**: Modern, accessible Jinja2 HTML templates adhering to healthcare design guidelines with responsive layouts.
10. **`static/css/style.css`**: Healthcare-themed styling with high color contrast, accessible tap targets (>= 48px), and modern card elevations.
11. **`static/js/main.js`**: Handles drag-and-drop image dropping, instant client-side preview, and AJAX submission with progressive status updates.

---

## 💻 Windows Setup Instructions (Step-by-Step)

Follow these simple steps in **PowerShell** or **Command Prompt** to run the project on Windows:

### Step 1: Open Terminal in the Project Folder
Open PowerShell or Command Prompt and navigate to the project directory:
```powershell
cd C:\Users\lavan\.gemini\antigravity\scratch\skin-disease-prediction
```

### Step 2: Create a Python Virtual Environment
Creating a virtual environment ensures project dependencies do not interfere with system packages:
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
- **On PowerShell**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If PowerShell displays an execution policy error, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then re-run the activate command).*
- **On Command Prompt (cmd.exe)**:
  ```cmd
  venv\Scripts\activate.bat
  ```
*(Once activated, you will see `(venv)` prefixed in your terminal prompt).*

### Step 4: Install Dependencies
Install all required libraries using pip:
```powershell
pip install -r requirements.txt
```

### Step 5: Start the Flask Application
Run the web application server:
```powershell
python app.py
```

You should see terminal output similar to:
```text
======================================================================
Starting Flask Web Application for Skin Disease Prediction
B.Tech Minor Project-I
Server accessible at: http://127.0.0.1:5000
======================================================================
 * Running on http://127.0.0.1:5000
```

### Step 6: Open the Web Application in Your Browser
Open any web browser (Google Chrome, Microsoft Edge, Firefox) and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔬 Benchmark Dataset: HAM10000

The application is configured around the **HAM10000** ("Human Against Machine with 10000 training images") benchmark dataset hosted on the ISIC Archive and Harvard Dataverse.

### Target Diagnostic Classes (7 Categories):
| Code | Diagnosis Name | Severity | Dataset Count | Percentage |
| :--- | :--- | :--- | :--- | :--- |
| `nv` | **Melanocytic Nevi** | Benign (Common Mole) | 6,705 | 66.95% |
| `mel` | **Melanoma** | Malignant (High Risk Cancer) | 1,113 | 11.11% |
| `bkl` | **Benign Keratosis** | Benign (Seborrheic Keratosis) | 1,099 | 10.97% |
| `bcc` | **Basal Cell Carcinoma** | Malignant (Common Skin Cancer) | 514 | 5.13% |
| `akiec` | **Actinic Keratoses** | Pre-cancerous (Solar Keratosis) | 327 | 3.27% |
| `vasc` | **Vascular Lesions** | Benign (Angiomas / Granulomas) | 142 | 1.42% |
| `df` | **Dermatofibroma** | Benign (Fibrous Nodule) | 115 | 1.15% |

---

## 📊 Evaluation & The "Accuracy Paradox"

A critical point for **Minor Project-I viva evaluations**:
- Because Melanocytic Nevi (`nv`) represents **67%** of the dataset, an untrained model predicting `nv` on every image would score **67% raw accuracy** while having a **0% sensitivity** for Melanoma.
- In medical diagnostics, missing cancer (**False Negative**) is clinically unacceptable.
- Therefore, this project prioritizes **Sensitivity / Recall**, **Specificity**, **Macro F1-Score**, and **Confusion Matrix** analysis over overall accuracy.

---

## 🚀 Transitioning to Phase 2 (Model Training)

When you are ready to train the model in the next phase:
1. Download HAM10000 images and place them into `dataset/train/`, `dataset/validation/`, and `dataset/test/` organized into class folders (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).
2. Train a transfer learning network (e.g., MobileNetV2 or EfficientNetB0) and save the trained weights as `model/skin_disease_model.h5`.
3. Restart the Flask app. The application will automatically detect the weights file and enable live softmax disease predictions!
