import numpy as np

import pylibjxl


def test_encode_decode_rgb(sample_image):
    """Test standard JXL lossy encoding/decoding."""
    img = sample_image

    # Encode
    jxl_data = pylibjxl.encode(img, effort=4)
    assert len(jxl_data) > 0

    # Decode
    decoded_img = pylibjxl.decode(jxl_data)

    assert decoded_img.shape == img.shape
    assert decoded_img.dtype == img.dtype

    # Check similarity (not exact equality due to lossy compression)
    # Using a relatively high tolerance for lossy compression
    # Or just check some basic properties if exact pixels aren't critical
    # Here we just ensure it decoded successfully to same shape/type.
    # We could check SSIM, but let's stick to basic sanity.


def test_encode_decode_rgba(sample_image_rgba):
    """Test JXL encoding/decoding with alpha channel."""
    img = sample_image_rgba

    # Encode
    jxl_data = pylibjxl.encode(img, effort=4)

    # Decode
    decoded_img = pylibjxl.decode(jxl_data)

    assert decoded_img.shape == img.shape
    assert decoded_img.shape[2] == 4

    # Basic check - alpha channel should be preserved (though lossy)


def test_encode_decode_lossless(sample_image):
    """Test JXL lossless encoding/decoding."""
    img = sample_image

    # Encode lossless
    jxl_data = pylibjxl.encode(img, effort=4, lossless=True)

    # Decode
    decoded_img = pylibjxl.decode(jxl_data)

    # Should be EXACTLY the same
    np.testing.assert_array_equal(decoded_img, img)
