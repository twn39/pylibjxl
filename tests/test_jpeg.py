import numpy as np
import pytest

import pylibjxl


def test_roundtrip_rgb(sample_image):
    img = sample_image
    data = pylibjxl.encode_jpeg(img, quality=90)
    assert isinstance(data, bytes)
    assert len(data) > 0
    # Check signature
    assert data.startswith(b"\xff\xd8")

    decoded = pylibjxl.decode_jpeg(data)
    assert decoded.shape == img.shape
    assert decoded.dtype == np.uint8


def test_encode_rgba(sample_image_rgba):
    img = sample_image_rgba
    data = pylibjxl.encode_jpeg(img, quality=100)
    decoded = pylibjxl.decode_jpeg(data)
    # Alpha channel is dropped for JPEG
    assert decoded.shape == (img.shape[0], img.shape[1], 3)


def test_invalid_input():
    with pytest.raises(TypeError):
        from typing import Any
        pylibjxl.encode_jpeg(b"not an array"  # type: ignore
                             )

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
async def test_async_jpeg(sample_image):
    img = sample_image
    data = await pylibjxl.encode_jpeg_async(img)
    assert len(data) > 0

    decoded = await pylibjxl.decode_jpeg_async(data)
    assert decoded.shape == img.shape
