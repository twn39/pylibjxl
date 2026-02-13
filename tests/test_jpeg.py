import numpy as np
import pytest
import pylibjxl

def test_roundtrip_rgb():
    np.random.seed(42)
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    data = pylibjxl.encode_jpeg(img, quality=90)
    assert isinstance(data, bytes)
    assert len(data) > 0
    # Check signature
    assert data.startswith(b"\xff\xd8")
    
    decoded = pylibjxl.decode_jpeg(data)
    assert decoded.shape == (128, 128, 3)
    assert decoded.dtype == np.uint8

def test_encode_rgba():
    np.random.seed(42)
    img = np.random.randint(0, 256, (64, 64, 4), dtype=np.uint8)
    data = pylibjxl.encode_jpeg(img, quality=100)
    decoded = pylibjxl.decode_jpeg(data)
    # Alpha channel is dropped for JPEG
    assert decoded.shape == (64, 64, 3)

def test_invalid_input():
    with pytest.raises(TypeError):
        pylibjxl.encode_jpeg(b"not an array")
    
    with pytest.raises(ValueError):
        # Wrong dimensions
        pylibjxl.encode_jpeg(np.zeros((10, 10), dtype=np.uint8))

    with pytest.raises(ValueError):
        # 5 channels
        pylibjxl.encode_jpeg(np.zeros((10, 10, 5), dtype=np.uint8))

def test_decode_invalid_data():
    with pytest.raises(RuntimeError):
        pylibjxl.decode_jpeg(b"invalid jpeg data")

@pytest.mark.asyncio
async def test_async_jpeg():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    data = await pylibjxl.encode_jpeg_async(img)
    assert len(data) > 0
    
    decoded = await pylibjxl.decode_jpeg_async(data)
    assert decoded.shape == (32, 32, 3)
