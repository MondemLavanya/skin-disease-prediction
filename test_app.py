"""
test_app.py
==============================================================================
Test suite for Flask routes, endpoints, and predictor integration
==============================================================================
"""

import io
from PIL import Image
import app

def run_tests():
    print("=" * 60)
    print("RUNNING FLASK APP & PREDICTOR VERIFICATION TESTS")
    print("=" * 60)

    client = app.app.test_client()

    # Test 1: GET routes
    routes = [
        '/',
        '/dataset',
        '/about',
        '/evaluation',
        '/api/dataset-status',
        '/api/model-status'
    ]
    for r in routes:
        res = client.get(r)
        assert res.status_code == 200, f"Route {r} failed with status {res.status_code}"
        print(f"[PASS] Route '{r}' returned 200 OK")

    # Test 2: POST /predict with no file
    res = client.post('/predict', data={})
    assert res.status_code == 400
    print("[PASS] POST /predict without file returned 400 Bad Request")

    # Test 3: POST /predict with invalid extension (.txt)
    res = client.post('/predict', data={'image': (io.BytesIO(b'hello text file'), 'test.txt')})
    assert res.status_code == 400
    print("[PASS] POST /predict with .txt file returned 400 Bad Request")

    # Test 4: POST /predict with valid JPG image
    img_bytes = io.BytesIO()
    Image.new('RGB', (350, 400), color=(140, 100, 80)).save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    res = client.post('/predict', data={'image': (img_bytes, 'sample_lesion.jpg')})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.get_json()

    assert data['preprocessing_summary']['tensor_shape'] == [1, 224, 224, 3], f"Unexpected tensor shape: {data['preprocessing_summary']['tensor_shape']}"
    assert data['preprocessing_summary']['validation_passed'] is True
    assert data['preprocessing_summary']['color_channels'] == 3

    if data.get('is_trained'):
        assert data['status'] == 'success'
        assert 'top_prediction' in data
        assert len(data['all_predictions']) == 7
        assert 'confidence_pct' in data['top_prediction']
        print("[PASS] POST /predict with JPG returned genuine model inference and top prediction")
    else:
        assert data['status'] == 'pending_model_training'
        assert data['is_trained'] is False
        assert len(data['target_classes']) == 7
        assert 'academic_note' in data
        print("[PASS] POST /predict returned valid preprocessed tensor shape (1, 224, 224, 3) and academic note")

    # Test 5: POST /predict with valid PNG image
    png_bytes = io.BytesIO()
    Image.new('RGB', (250, 250), color=(180, 120, 90)).save(png_bytes, format='PNG')
    png_bytes.seek(0)

    res_png = client.post('/predict', data={'image': (png_bytes, 'sample_lesion.png')})
    assert res_png.status_code == 200
    data_png = res_png.get_json()
    if data_png.get('is_trained'):
        assert data_png['status'] == 'success'
        assert 'top_prediction' in data_png
        print("[PASS] POST /predict with PNG image succeeded with genuine model inference")
    else:
        assert data_png['status'] == 'pending_model_training'
        print("[PASS] POST /predict with PNG image succeeded with valid preprocessing")

    # Test 6: API Dataset status
    res_ds = client.get('/api/dataset-status')
    assert res_ds.status_code == 200
    ds_data = res_ds.get_json()
    assert 'structure' in ds_data and 'train' in ds_data and 'validation' in ds_data and 'test' in ds_data
    print("[PASS] GET /api/dataset-status returned live filesystem stats")

    # Test 7: API Model status
    res_model = client.get('/api/model-status')
    assert res_model.status_code == 200
    model_data = res_model.get_json()
    assert len(model_data['classes']) == 7
    if model_data.get('weights_exist'):
        assert model_data['is_loaded'] is True
        print("[PASS] GET /api/model-status correctly reported trained state with 7 classes")
    else:
        assert model_data['is_loaded'] is False
        print("[PASS] GET /api/model-status correctly reported un-trained state with 7 classes")

    # Test 8: Missing-model scenario verification (Academic Integrity Guard)
    from unittest.mock import patch
    with patch('predictor.load_trained_model', return_value=None):
        fallback_bytes = io.BytesIO()
        Image.new('RGB', (224, 224), color=(100, 100, 100)).save(fallback_bytes, format='JPEG')
        fallback_bytes.seek(0)
        res_fallback = client.post('/predict', data={'image': (fallback_bytes, 'missing_model_test.jpg')})
        assert res_fallback.status_code == 200
        fallback_data = res_fallback.get_json()
        assert fallback_data['status'] == 'pending_model_training'
        assert fallback_data['is_trained'] is False
        assert 'academic_note' in fallback_data
        assert len(fallback_data['target_classes']) == 7
        print("[PASS] Missing-model scenario correctly triggers academic integrity fallback")

    print("\n" + "=" * 60)
    print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    run_tests()
