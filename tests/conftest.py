import pytest
from pathlib import Path
import pylibjxl
import numpy as np

@pytest.fixture(scope="session")
def real_image_path():
    """Path to the real test image."""
    base_dir = Path(__file__).parent.parent
    path = base_dir / "images" / "test.jpg"
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return path

@pytest.fixture(scope="session")
def real_image_bytes(real_image_path):
    """Bytes of the real test image."""
    return real_image_path.read_bytes()

@pytest.fixture(scope="session")
def sample_image(real_image_bytes):
    """Decoded numpy array of the test image (RGB)."""
    return pylibjxl.decode_jpeg(real_image_bytes)

@pytest.fixture(scope="session")
def sample_image_rgba(sample_image):
    """Decoded numpy array of the test image with Alpha channel (RGBA)."""
    # Create an alpha channel (255 fully opaque)
    h, w, c = sample_image.shape
    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    return np.concatenate([sample_image, alpha], axis=2)
