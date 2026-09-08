import numpy as np
import pytest

import pylibjxl


@pytest.fixture
def gray_2d():
    """Create a 2D grayscale gradient image (H, W)."""
    h, w = 64, 48
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    return ((y + x) // 2).astype(np.uint8)


@pytest.fixture
def gray_3d(gray_2d):
    """Create a 3D grayscale image with 1 channel (H, W, 1)."""
    return gray_2d[:, :, None]


def test_jxl_2d_roundtrip_lossless(gray_2d):
    """Test lossless JXL encode/decode for 2D array."""
    data = pylibjxl.encode(gray_2d, lossless=True)
    assert isinstance(data, bytes)
    assert len(data) > 0

    decoded = pylibjxl.decode(data)
    assert decoded.shape == gray_2d.shape
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, gray_2d)


def test_jxl_2d_roundtrip_lossy(gray_2d):
    """Test lossy JXL encode/decode for 2D array."""
    data = pylibjxl.encode(gray_2d, lossless=False, distance=1.0)
    assert isinstance(data, bytes)
    assert len(data) > 0

    decoded = pylibjxl.decode(data)
    assert decoded.shape == gray_2d.shape
    assert decoded.dtype == np.uint8
    # PSNR/MAE check: lossy compression on gradient should be very close
    mae = np.mean(np.abs(decoded.astype(float) - gray_2d.astype(float)))
    assert mae < 5.0


def test_jxl_3d_roundtrip(gray_3d, gray_2d):
    """Test JXL encode with (H, W, 1) and decode."""
    data = pylibjxl.encode(gray_3d, lossless=True)
    assert isinstance(data, bytes)

    # Without out: decodes to 2D standard
    decoded = pylibjxl.decode(data)
    assert decoded.shape == gray_2d.shape
    np.testing.assert_array_equal(decoded, gray_2d)


def test_jxl_grayscale_inplace_out(gray_2d):
    """Test JXL decode with in-place pre-allocated 2D and 3D out buffers."""
    data = pylibjxl.encode(gray_2d, lossless=True)

    # 2D out buffer
    out_2d = np.zeros(gray_2d.shape, dtype=np.uint8)
    ret_2d = pylibjxl.decode(data, out=out_2d)
    assert ret_2d is out_2d
    np.testing.assert_array_equal(out_2d, gray_2d)

    # 3D (H, W, 1) out buffer
    out_3d = np.zeros((gray_2d.shape[0], gray_2d.shape[1], 1), dtype=np.uint8)
    ret_3d = pylibjxl.decode(data, out=out_3d)
    assert ret_3d is out_3d
    np.testing.assert_array_equal(out_3d[:, :, 0], gray_2d)


def test_jpeg_2d_roundtrip(gray_2d):
    """Test JPEG encode/decode for 2D array."""
    data = pylibjxl.encode_jpeg(gray_2d, quality=95)
    assert isinstance(data, bytes)
    assert data.startswith(b"\xff\xd8")

    decoded = pylibjxl.decode_jpeg(data)
    assert decoded.shape == gray_2d.shape
    assert decoded.dtype == np.uint8

    mae = np.mean(np.abs(decoded.astype(float) - gray_2d.astype(float)))
    assert mae < 5.0


def test_jpeg_3d_roundtrip(gray_3d, gray_2d):
    """Test JPEG encode with (H, W, 1) and decode."""
    data = pylibjxl.encode_jpeg(gray_3d, quality=95)
    assert isinstance(data, bytes)
    assert data.startswith(b"\xff\xd8")

    decoded = pylibjxl.decode_jpeg(data)
    assert decoded.shape == gray_2d.shape
    assert decoded.dtype == np.uint8


def test_jpeg_grayscale_inplace_out(gray_2d):
    """Test JPEG decode with 2D, 3D (1 channel), and 3D (3 channel RGB) out buffers."""
    data = pylibjxl.encode_jpeg(gray_2d, quality=95)

    # 2D out
    out_2d = np.zeros(gray_2d.shape, dtype=np.uint8)
    ret_2d = pylibjxl.decode_jpeg(data, out=out_2d)
    assert ret_2d is out_2d

    # 3D (H, W, 1) out
    out_3d = np.zeros((gray_2d.shape[0], gray_2d.shape[1], 1), dtype=np.uint8)
    ret_3d = pylibjxl.decode_jpeg(data, out=out_3d)
    assert ret_3d is out_3d
    np.testing.assert_array_equal(out_3d[:, :, 0], out_2d)

    # 3D (H, W, 3) out (decompress grayscale to RGB)
    out_rgb = np.zeros((gray_2d.shape[0], gray_2d.shape[1], 3), dtype=np.uint8)
    ret_rgb = pylibjxl.decode_jpeg(data, out=out_rgb)
    assert ret_rgb is out_rgb
    np.testing.assert_array_equal(out_rgb[:, :, 0], out_2d)
    np.testing.assert_array_equal(out_rgb[:, :, 1], out_2d)
    np.testing.assert_array_equal(out_rgb[:, :, 2], out_2d)


def test_file_io_grayscale(tmp_path, gray_2d):
    """Test write and read with 2D grayscale."""
    file_path = tmp_path / "gray.jxl"

    pylibjxl.write(file_path, gray_2d, lossless=True)
    assert file_path.exists()

    # Normal read
    read_img = pylibjxl.read(file_path)
    assert read_img.shape == gray_2d.shape
    np.testing.assert_array_equal(read_img, gray_2d)

    # Read with mmap
    read_mmap = pylibjxl.read(file_path, use_mmap=True)
    np.testing.assert_array_equal(read_mmap, gray_2d)

    # Read with out
    out = np.zeros_like(gray_2d)
    read_out = pylibjxl.read(file_path, out=out)
    assert read_out is out
    np.testing.assert_array_equal(read_out, gray_2d)


@pytest.mark.asyncio
async def test_async_grayscale(gray_2d):
    """Test async encode/decode for grayscale."""
    jxl_data = await pylibjxl.encode_async(gray_2d, lossless=True)
    decoded_jxl = await pylibjxl.decode_async(jxl_data)
    np.testing.assert_array_equal(decoded_jxl, gray_2d)

    jpeg_data = await pylibjxl.encode_jpeg_async(gray_2d)
    decoded_jpeg = await pylibjxl.decode_jpeg_async(jpeg_data)
    assert decoded_jpeg.shape == gray_2d.shape


def test_context_grayscale(gray_2d):
    """Test JXL context manager with grayscale images."""
    with pylibjxl.JXL(lossless=True) as jxl:
        data = jxl.encode(gray_2d)
        decoded = jxl.decode(data)
        np.testing.assert_array_equal(decoded, gray_2d)

        jpeg = jxl.encode_jpeg(gray_2d)
        decoded_jpeg = jxl.decode_jpeg(jpeg)
        assert decoded_jpeg.shape == gray_2d.shape


def test_invalid_grayscale_dimensions():
    """Test error handling for unsupported dimensions."""
    # 1D array
    with pytest.raises(ValueError):
        pylibjxl.encode(np.zeros((50,), dtype=np.uint8))

    with pytest.raises(ValueError):
        pylibjxl.encode_jpeg(np.zeros((50,), dtype=np.uint8))

    # 4D array
    with pytest.raises(ValueError):
        pylibjxl.encode(np.zeros((10, 10, 1, 1), dtype=np.uint8))

    with pytest.raises(ValueError):
        pylibjxl.encode_jpeg(np.zeros((10, 10, 1, 1), dtype=np.uint8))

    # 3D array with 2 channels
    with pytest.raises(ValueError):
        pylibjxl.encode(np.zeros((10, 10, 2), dtype=np.uint8))

    with pytest.raises(ValueError):
        pylibjxl.encode_jpeg(np.zeros((10, 10, 2), dtype=np.uint8))
