import numpy as np
import pytest
import pylibjxl


def _make_rgb(w=64, h=64):
    np.random.seed(42)
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


# ─── JPEG File I/O ──────────────────────────────────────────────────────────────


class TestJpegFileIO:
    def test_write_read_roundtrip(self, tmp_path):
        img = _make_rgb()
        path = tmp_path / "test.jpg"
        pylibjxl.write_jpeg(path, img, quality=100)
        assert path.exists()
        result = pylibjxl.read_jpeg(path)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_write_creates_parent_dirs(self, tmp_path):
        img = _make_rgb()
        path = tmp_path / "a" / "b" / "test.jpeg"
        pylibjxl.write_jpeg(path, img)
        assert path.exists()

    def test_read_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            pylibjxl.read_jpeg("/nonexistent/path.jpg")

    def test_write_rgba(self, tmp_path):
        img = np.random.randint(0, 256, (32, 32, 4), dtype=np.uint8)
        path = tmp_path / "rgba.jpg"
        pylibjxl.write_jpeg(path, img)
        result = pylibjxl.read_jpeg(path)
        # Alpha is dropped by JPEG
        assert result.shape == (32, 32, 3)


class TestAsyncJpegFileIO:
    async def test_async_write_read(self, tmp_path):
        img = _make_rgb()
        path = tmp_path / "async.jpg"
        await pylibjxl.write_jpeg_async(path, img)
        assert path.exists()
        result = await pylibjxl.read_jpeg_async(path)
        assert result.shape == (64, 64, 3)


# ─── Cross-Format Conversion ────────────────────────────────────────────────────


class TestCrossFormatConversion:
    def test_jpeg_to_jxl_conversion(self, tmp_path):
        """JPEG file → JXL file (lossless transcoding)."""
        img = _make_rgb()
        jpeg_path = tmp_path / "input.jpg"
        jxl_path = tmp_path / "output.jxl"
        pylibjxl.write_jpeg(jpeg_path, img)
        pylibjxl.convert_jpeg_to_jxl(jpeg_path, jxl_path)
        assert jxl_path.exists()
        # The JXL should be readable as a JXL image
        jxl_data = jxl_path.read_bytes()
        assert len(jxl_data) > 0

    def test_jxl_to_jpeg_conversion(self, tmp_path):
        """JXL file → JPEG file (lossless reconstruction)."""
        img = _make_rgb()
        jpeg_path = tmp_path / "input.jpg"
        jxl_path = tmp_path / "intermediate.jxl"
        output_jpeg_path = tmp_path / "output.jpg"
        # Create JPEG → transcode to JXL → convert back to JPEG
        pylibjxl.write_jpeg(jpeg_path, img)
        pylibjxl.convert_jpeg_to_jxl(jpeg_path, jxl_path)
        pylibjxl.convert_jxl_to_jpeg(jxl_path, output_jpeg_path)
        assert output_jpeg_path.exists()
        # The reconstructed JPEG should be identical to the original
        original_data = jpeg_path.read_bytes()
        restored_data = output_jpeg_path.read_bytes()
        assert original_data == restored_data

    def test_jpeg_to_jxl_creates_dirs(self, tmp_path):
        img = _make_rgb()
        jpeg_path = tmp_path / "input.jpg"
        jxl_path = tmp_path / "sub" / "dir" / "output.jxl"
        pylibjxl.write_jpeg(jpeg_path, img)
        pylibjxl.convert_jpeg_to_jxl(jpeg_path, jxl_path)
        assert jxl_path.exists()

    def test_convert_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            pylibjxl.convert_jpeg_to_jxl("/nonexistent.jpg", tmp_path / "out.jxl")
        with pytest.raises(FileNotFoundError):
            pylibjxl.convert_jxl_to_jpeg("/nonexistent.jxl", tmp_path / "out.jpg")

    def test_jxl_to_jpeg_non_jpeg_origin_raises(self, tmp_path):
        """Converting a non-JPEG-origin JXL should raise RuntimeError."""
        img = _make_rgb()
        jxl_path = tmp_path / "pixel.jxl"
        jpeg_path = tmp_path / "out.jpg"
        # Create a JXL from raw pixels (not from JPEG transcoding)
        pylibjxl.write(jxl_path, img, effort=4)
        with pytest.raises(RuntimeError, match="reconstructible"):
            pylibjxl.convert_jxl_to_jpeg(jxl_path, jpeg_path)


class TestAsyncCrossFormatConversion:
    async def test_async_jpeg_to_jxl(self, tmp_path):
        img = _make_rgb()
        jpeg_path = tmp_path / "input.jpg"
        jxl_path = tmp_path / "output.jxl"
        pylibjxl.write_jpeg(jpeg_path, img)
        await pylibjxl.convert_jpeg_to_jxl_async(jpeg_path, jxl_path)
        assert jxl_path.exists()

    async def test_async_jxl_to_jpeg(self, tmp_path):
        img = _make_rgb()
        jpeg_path = tmp_path / "input.jpg"
        jxl_path = tmp_path / "intermediate.jxl"
        output_path = tmp_path / "output.jpg"
        pylibjxl.write_jpeg(jpeg_path, img)
        await pylibjxl.convert_jpeg_to_jxl_async(jpeg_path, jxl_path)
        await pylibjxl.convert_jxl_to_jpeg_async(jxl_path, output_path)
        assert output_path.exists()
        assert jpeg_path.read_bytes() == output_path.read_bytes()
