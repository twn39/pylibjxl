import pylibjxl
import numpy as np
import pytest

def test_encode_decode_rgb():
    # Create a simple RGB image
    width, height = 100, 100
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :50, 0] = 255  # Red left half
    img[:, 50:, 1] = 255  # Green right half
    
    # Encode
    jxl_data = pylibjxl.encode(img, effort=4)
    assert len(jxl_data) > 0
    
    # Decode
    decoded_img = pylibjxl.decode(jxl_data)
    
    assert decoded_img.shape == img.shape
    assert decoded_img.dtype == img.dtype
    
    # Check some pixels (note: JXL lossy might not be exact)
    # With distance=1.0 it should be close.
    # For lossless, we could set distance=0 and use lossless=True but I haven't exposed lossless=True yet.
    # Actually I set distance=1.0 by default.
    
    np.testing.assert_allclose(decoded_img[:, :50, 0], 255, atol=10)
    np.testing.assert_allclose(decoded_img[:, 50:, 1], 255, atol=10)

def test_encode_decode_rgba():
    # Create a simple RGBA image
    width, height = 100, 100
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[:, :50, 0] = 255  # Red left
    img[:, :50, 3] = 255  # Opaque left
    img[:, 50:, 1] = 255  # Green right
    img[:, 50:, 3] = 128  # Half-transparent right
    
    # Encode
    jxl_data = pylibjxl.encode(img, effort=4)
    
    # Decode
    decoded_img = pylibjxl.decode(jxl_data)
    
    assert decoded_img.shape == img.shape
    assert decoded_img.shape[2] == 4
    
    np.testing.assert_allclose(decoded_img[:, :50, 0], 255, atol=10)
    np.testing.assert_allclose(decoded_img[:, 50:, 1], 255, atol=10)
    np.testing.assert_allclose(decoded_img[:, :50, 3], 255, atol=10)
    np.testing.assert_allclose(decoded_img[:, 50:, 3], 128, atol=10)

def test_encode_decode_lossless():
    # Create a random image
    width, height = 50, 50
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    # Encode lossless
    jxl_data = pylibjxl.encode(img, effort=4, lossless=True)
    
    # Decode
    decoded_img = pylibjxl.decode(jxl_data)
    
    # Should be EXACTLY the same
    np.testing.assert_array_equal(decoded_img, img)
