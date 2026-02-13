import numpy as np
import pytest
import pylibjxl

def test_transcode_roundtrip():
    # 1. Create source JPEG using generic codec
    np.random.seed(123)
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    jpeg_data = pylibjxl.encode_jpeg(img, quality=95)
    
    # 2. Transcode JPEG -> JXL (lossless recompression)
    jxl_data = pylibjxl.jpeg_to_jxl(jpeg_data)
    assert len(jxl_data) < len(jpeg_data) # Usually smaller, but depends on content. Random noise might not compress well.
    # Actually random noise compresses poorly in both.
    # Let's check it's valid JXL at least.
    assert jxl_data.startswith(b"\xff\x0a") or jxl_data.startswith(b"\x00\x00\x00\x0cJXL ")

    # 3. Transcode JXL -> JPEG (reconstruction)
    restored_jpeg = pylibjxl.jxl_to_jpeg(jxl_data)
    
    # 4. Verify bit-exactness with original JPEG
    assert jpeg_data == restored_jpeg

def test_transcode_invalid_jpeg():
    with pytest.raises(RuntimeError):
        pylibjxl.jpeg_to_jxl(b"not a jpeg")

def test_transcode_non_reconstructible_jxl():
    # Create a JXL from pixels (not from JPEG)
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    jxl_data = pylibjxl.encode(img)
    
    # Attempt to reconstruct JPEG (should fail as no JPEG codestream is embedded)
    with pytest.raises(RuntimeError, match="reconstructible"):
        pylibjxl.jxl_to_jpeg(jxl_data)

@pytest.mark.asyncio
async def test_transcode_async():
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    jpeg_data = await pylibjxl.encode_jpeg_async(img, quality=90)
    
    jxl_data = await pylibjxl.jpeg_to_jxl_async(jpeg_data)
    restored = await pylibjxl.jxl_to_jpeg_async(jxl_data)
    
    assert jpeg_data == restored
