/**
 * main.js
 * ==============================================================================
 * Skin Disease Prediction Using Deep Learning
 * Client-Side Controller: Upload, Drag & Drop, Preview & AJAX Inference
 * ==============================================================================
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('image-file-input');
    const uploadForm = document.getElementById('lesion-upload-form');
    const predictBtn = document.getElementById('predict-btn');
    const resetBtn = document.getElementById('reset-btn');
    const btnRemovePreview = document.getElementById('btn-remove-preview');

    // Preview elements
    const emptyState = document.getElementById('dropzone-empty-state');
    const previewContainer = document.getElementById('image-preview-container');
    const previewImage = document.getElementById('preview-image');
    const previewFilename = document.getElementById('preview-filename');
    const previewSize = document.getElementById('preview-size');
    const previewDims = document.getElementById('preview-dims');

    // Loading & Error elements
    const loadingIndicator = document.getElementById('loading-indicator');
    const step1 = document.getElementById('step-1');
    const step2 = document.getElementById('step-2');
    const step3 = document.getElementById('step-3');
    const errorAlert = document.getElementById('error-alert');
    const errorMessage = document.getElementById('error-message');

    // Results container elements
    const resultsEmptyState = document.getElementById('results-empty-state');
    const resultsContent = document.getElementById('results-content');
    const resultStatusBanner = document.getElementById('result-status-banner');

    // Diagnostic elements
    const diagShape = document.getElementById('diag-shape');
    const diagChannels = document.getElementById('diag-channels');
    const diagTargetRes = document.getElementById('diag-target-res');
    const diagNormRange = document.getElementById('diag-norm-range');
    const diagMean = document.getElementById('diag-mean');
    const diagStatus = document.getElementById('diag-status');

    // Result views
    const pretrainingNotice = document.getElementById('pretraining-notice');
    const pretrainingAcademicText = document.getElementById('pretraining-academic-text');
    const inferenceResults = document.getElementById('inference-results');
    const topDiseaseName = document.getElementById('top-disease-name');
    const topDiseaseType = document.getElementById('top-disease-type');
    const topDiseaseConf = document.getElementById('top-disease-conf');
    const topConfidenceBar = document.getElementById('top-confidence-bar');
    const probabilitiesList = document.getElementById('probabilities-list');

    // Allowed file formats
    const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png'];
    const MAX_FILE_SIZE_BYTES = 16 * 1024 * 1024; // 16 MB

    let selectedFile = null;

    // ==========================================================================
    // 1. Dropzone & File Selection Handlers
    // ==========================================================================

    // Click on dropzone opens native file dialog
    if (dropzone && fileInput) {
        dropzone.addEventListener('click', (e) => {
            // Prevent re-trigger if clicking change image button
            if (e.target.closest('#btn-remove-preview')) return;
            fileInput.click();
        });

        // Keyboard accessibility (Space or Enter on focused dropzone)
        dropzone.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                fileInput.click();
            }
        });

        // Drag & Drop visual feedback
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('dragover');
            });
        });

        // Drop event
        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files.length > 0) {
                handleFileSelection(dt.files[0]);
            }
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelection(e.target.files[0]);
            }
        });
    }

    if (btnRemovePreview) {
        btnRemovePreview.addEventListener('click', (e) => {
            e.stopPropagation();
            resetForm();
        });
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            resetForm();
        });
    }

    // ==========================================================================
    // 2. Client-Side File Validation & Preview
    // ==========================================================================

    function handleFileSelection(file) {
        hideError();

        // 1. Validate file extension
        const fileName = file.name.toLowerCase();
        const isValidExtension = ALLOWED_EXTENSIONS.some(ext => fileName.endsWith(ext));
        if (!isValidExtension) {
            showError(`Unsupported file format. Please upload a JPG, JPEG, or PNG image.`);
            return;
        }

        // 2. Validate file size
        if (file.size > MAX_FILE_SIZE_BYTES) {
            showError(`File size exceeds 16 MB limit (${(file.size / (1024 * 1024)).toFixed(1)} MB).`);
            return;
        }

        selectedFile = file;

        // 3. Read image for preview and dimensions
        const reader = new FileReader();
        reader.onload = (e) => {
            const dataUrl = e.target.result;
            previewImage.src = dataUrl;

            // Load into offscreen Image to compute dimensions
            const tempImg = new Image();
            tempImg.onload = () => {
                previewDims.textContent = `${tempImg.naturalWidth} × ${tempImg.naturalHeight} px`;
            };
            tempImg.src = dataUrl;

            // Update UI metadata
            previewFilename.textContent = file.name;
            previewSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;

            // Switch to preview view
            emptyState.style.display = 'none';
            previewContainer.style.display = 'flex';
            predictBtn.disabled = false;
        };

        reader.readAsDataURL(file);
    }

    function resetForm() {
        selectedFile = null;
        if (fileInput) fileInput.value = '';
        if (previewImage) previewImage.src = '';
        
        emptyState.style.display = 'block';
        previewContainer.style.display = 'none';
        predictBtn.disabled = true;

        hideError();
        hideLoading();

        // Reset results to placeholder
        if (resultsContent) resultsContent.style.display = 'none';
        if (resultsEmptyState) resultsEmptyState.style.display = 'block';
    }

    // ==========================================================================
    // 3. AJAX Submission & Inference Orchestration
    // ==========================================================================

    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            if (!selectedFile) {
                showError("Please select or drop a skin lesion image first.");
                return;
            }

            hideError();
            showLoading();

            const formData = new FormData();
            formData.append('image', selectedFile);

            try {
                // Simulate progressive loading steps for clear visual comprehension
                updateLoadingStep(1);
                await delay(300);
                updateLoadingStep(2);
                await delay(400);
                updateLoadingStep(3);

                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                hideLoading();

                if (!response.ok || data.status === 'error') {
                    showError(data.message || 'An error occurred during prediction.');
                    return;
                }

                renderResults(data);

            } catch (err) {
                hideLoading();
                showError(`Network connection error: ${err.message}. Please check if the Flask server is running.`);
            }
        });
    }

    // ==========================================================================
    // 4. Render Preprocessing Diagnostics & Results
    // ==========================================================================

    function renderResults(data) {
        if (!resultsContent || !resultsEmptyState) return;

        resultsEmptyState.style.display = 'none';
        resultsContent.style.display = 'block';

        // 1. Populate Preprocessing Diagnostics
        if (data.preprocessing_summary) {
            const p = data.preprocessing_summary;
            diagShape.textContent = JSON.stringify(p.tensor_shape);
            diagChannels.textContent = `${p.color_channels} Channels (RGB)`;
            diagTargetRes.textContent = p.target_dimensions;
            diagNormRange.textContent = p.normalization_range;
            diagMean.textContent = p.pixel_mean !== undefined ? p.pixel_mean.toFixed(3) : '-';
            diagStatus.textContent = p.validation_passed ? 'Verified ✓' : 'Failed ✕';
        }

        // 2. Branch: Pre-training Mode (Weights Pending) vs Genuine Model Inference
        if (data.status === 'pending_model_training' || !data.is_trained) {
            // Render Pre-training verification banner
            resultStatusBanner.className = 'result-status-banner banner-verified';
            resultStatusBanner.innerHTML = `
                <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                    <polyline points="22 4 12 14.01 9 11.01"/>
                </svg>
                <span>Image Preprocessing Verified • Model Weights Pending (Phase 2)</span>
            `;

            if (pretrainingAcademicText && data.academic_note) {
                pretrainingAcademicText.textContent = data.academic_note;
            }

            pretrainingNotice.style.display = 'block';
            inferenceResults.style.display = 'none';

        } else if (data.status === 'success' && data.is_trained) {
            // Model inference succeeded
            resultStatusBanner.className = 'result-status-banner banner-inference';
            resultStatusBanner.innerHTML = `
                <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <polyline points="12 6 12 12 14 14"/>
                </svg>
                <span>Deep Learning Inference Completed Successfully</span>
            `;

            pretrainingNotice.style.display = 'none';
            inferenceResults.style.display = 'block';

            // Top Prediction
            if (data.top_prediction) {
                const top = data.top_prediction;
                topDiseaseName.textContent = top.name;
                topDiseaseType.textContent = top.type;
                topDiseaseType.className = `badge-pill badge-${(top.type || '').toLowerCase().replace(/[^a-z0-9]/g, '-')}`;
                topDiseaseConf.textContent = `${top.confidence_pct}%`;
                topConfidenceBar.style.width = `${top.confidence_pct}%`;
            }

            // Probability distribution breakdown
            if (data.all_predictions && probabilitiesList) {
                probabilitiesList.innerHTML = '';
                data.all_predictions.forEach(item => {
                    const row = document.createElement('div');
                    row.className = 'prob-row';
                    row.innerHTML = `
                        <span class="prob-name" title="${item.name}">${item.name}</span>
                        <div class="prob-bar-container">
                            <div class="prob-bar" style="width: ${item.confidence_pct}%"></div>
                        </div>
                        <span class="prob-pct">${item.confidence_pct}%</span>
                    `;
                    probabilitiesList.appendChild(row);
                });
            }
        }
    }

    // ==========================================================================
    // 5. Loading, Step Animation, & Error Helpers
    // ==========================================================================

    function showLoading() {
        if (loadingIndicator) loadingIndicator.style.display = 'flex';
        if (predictBtn) predictBtn.disabled = true;
    }

    function hideLoading() {
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        if (predictBtn && selectedFile) predictBtn.disabled = false;
    }

    function updateLoadingStep(stepNum) {
        [step1, step2, step3].forEach((el, idx) => {
            if (!el) return;
            if (idx + 1 === stepNum) {
                el.classList.add('active');
            } else if (idx + 1 < stepNum) {
                el.classList.remove('active');
                el.style.color = '#059669';
                el.textContent = el.textContent.replace('...', ' ✓');
            }
        });
    }

    function showError(msg) {
        if (errorAlert && errorMessage) {
            errorMessage.textContent = msg;
            errorAlert.style.display = 'flex';
        }
    }

    function hideError() {
        if (errorAlert) {
            errorAlert.style.display = 'none';
        }
    }

    function delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
});
