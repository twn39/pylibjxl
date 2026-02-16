from pathlib import Path

import numpy as np
import pytest

import pylibjxl


@pytest.fixture(scope="session")
def real_image_path():
    """Path to the real test image."""
    print("\nDEBUG: Initializing real_image_path fixture...")
    base_dir = Path(__file__).parent.parent
    path = base_dir / "images" / "test.jpg"
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    print(f"DEBUG: Found test image at {path}")
    return path


@pytest.fixture(scope="session")
def real_image_bytes(real_image_path):
    """Bytes of the real test image."""
    print("DEBUG: Initializing real_image_bytes fixture...")
    data = real_image_path.read_bytes()
    print(f"DEBUG: Read {len(data)} bytes from test image")
    return data


@pytest.fixture(scope="session")
def sample_image(real_image_bytes):
    """Decoded numpy array of the test image (RGB)."""
    print("DEBUG: Initializing sample_image fixture (decode_jpeg)...")
    img = pylibjxl.decode_jpeg(real_image_bytes)
    print(f"DEBUG: Decoded image shape: {img.shape}")
    return img


@pytest.fixture(scope="session")
def sample_image_rgba(sample_image):
    """Decoded numpy array of the test image with Alpha channel (RGBA)."""
    print("DEBUG: Initializing sample_image_rgba fixture...")
    # Create an alpha channel (255 fully opaque)
    h, w, c = sample_image.shape
    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    res = np.concatenate([sample_image, alpha], axis=2)
    print(f"DEBUG: Created RGBA image shape: {res.shape}")
    return res
