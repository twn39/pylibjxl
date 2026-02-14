import numpy as np
import pytest

import pylibjxl

# ─── Free Function: read/write ─────────────────────────────────────────────────


class TestReadWrite:
    def test_write_read_roundtrip(self, tmp_path, sample_image):
        img = sample_image
        path = tmp_path / "test.jxl"
        pylibjxl.write(path, img, effort=4, lossless=True)
        assert path.exists()
        result = pylibjxl.read(path)
        np.testing.assert_array_equal(result, img)

    def test_write_read_rgba(self, tmp_path, sample_image_rgba):
        img = sample_image_rgba
        path = tmp_path / "test_rgba.jxl"
        pylibjxl.write(path, img, effort=4, lossless=True)
        result = pylibjxl.read(path)
        # JXL lossless should preserve exact values including alpha
        np.testing.assert_array_equal(result, img)

    def test_write_creates_parent_dirs(self, tmp_path, sample_image):
        img = sample_image
        path = tmp_path / "a" / "b" / "c" / "test.jxl"
        pylibjxl.write(path, img, effort=4, lossless=True)
        assert path.exists()
        result = pylibjxl.read(path)
        np.testing.assert_array_equal(result, img)

    def test_read_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            pylibjxl.read("/nonexistent/path.jxl")

    def test_write_lossy(self, tmp_path, sample_image):
        img = sample_image
        path = tmp_path / "lossy.jxl"
        pylibjxl.write(path, img, effort=4, distance=1.0)
        result = pylibjxl.read(path)
        assert result.dtype == np.uint8
        # Lossy compression: exact values will differ, but shape/dtype should match


# ─── Context Manager: read/write ──────────────────────────────────────────────


class TestJXLReadWrite:
    def test_context_write_read(self, tmp_path, sample_image):
        img = sample_image
        path = tmp_path / "ctx.jxl"
        with pylibjxl.JXL(effort=4) as jxl:
            jxl.write(path, img, lossless=True)
            result = jxl.read(path)
            np.testing.assert_array_equal(result, img)

    def test_context_multiple_files(self, tmp_path, sample_image):
        img1 = sample_image
        img2 = np.ascontiguousarray(np.flipud(sample_image))
        p1 = tmp_path / "img1.jxl"
        p2 = tmp_path / "img2.jxl"
        with pylibjxl.JXL(effort=4) as jxl:
            jxl.write(p1, img1, lossless=True)
            jxl.write(p2, img2, lossless=True)
            r1 = jxl.read(p1)
            r2 = jxl.read(p2)
            np.testing.assert_array_equal(r1, img1)
            np.testing.assert_array_equal(r2, img2)

    def test_closed_read_raises(self, tmp_path, sample_image):
        img = sample_image
        path = tmp_path / "closed.jxl"
        pylibjxl.write(path, img, effort=4, lossless=True)
        jxl = pylibjxl.JXL()
        jxl.close()
        with pytest.raises(RuntimeError, match="closed"):
            jxl.read(path)

    def test_closed_write_raises(self, tmp_path, sample_image):
        jxl = pylibjxl.JXL()
        jxl.close()
        with pytest.raises(RuntimeError, match="closed"):
            jxl.write(tmp_path / "fail.jxl", sample_image)


# ─── Async Free Function ───────────────────────────────────────────────────────


class TestAsyncReadWrite:
    @pytest.mark.asyncio
    async def test_async_write_read_roundtrip(self, tmp_path, sample_image):
        img = sample_image
        path = tmp_path / "async.jxl"
        await pylibjxl.write_async(path, img, effort=4, lossless=True)
        assert path.exists()
        result = await pylibjxl.read_async(path)
        np.testing.assert_array_equal(result, img)


# ─── Async Context Manager ─────────────────────────────────────────────────────


class TestAsyncJXLReadWrite:
    @pytest.mark.asyncio
    async def test_async_context_write_read(self, tmp_path, sample_image):
        img = sample_image
        path = tmp_path / "async_ctx.jxl"
        async with pylibjxl.AsyncJXL(effort=4) as jxl:
            await jxl.write_async(path, img, lossless=True)
            result = await jxl.read_async(path)
            np.testing.assert_array_equal(result, img)
        assert jxl.closed
