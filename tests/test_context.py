import numpy as np
import pylibjxl
import pytest

class TestJXL:
    def test_encode_decode_roundtrip(self, sample_image):
        """Encode and decode within the same context."""
        img = sample_image
        with pylibjxl.JXL(effort=4) as jxl:
            data = jxl.encode(img, lossless=True)
            result = jxl.decode(data)
            np.testing.assert_array_equal(result, img)

    def test_context_lifecycle(self):
        with pylibjxl.JXL() as jxl:
            assert not jxl.closed
        assert jxl.closed

    def test_multiple_operations(self, sample_image):
        img1 = sample_image
        # Create a second image by flipping the first one
        img2 = np.ascontiguousarray(np.flipud(sample_image))
        
        with pylibjxl.JXL(effort=4) as jxl:
            d1 = jxl.encode(img1, lossless=True)
            d2 = jxl.encode(img2, lossless=True)
            r1 = jxl.decode(d1)
            r2 = jxl.decode(d2)
            np.testing.assert_array_equal(r1, img1)
            np.testing.assert_array_equal(r2, img2)

    def test_per_call_override(self, sample_image):
        img = sample_image
        with pylibjxl.JXL(effort=4, distance=1.0) as jxl:
            lossy = jxl.encode(img)
            lossless = jxl.encode(img, lossless=True)
            assert len(lossy) > 0
            assert len(lossless) > 0

    def test_closed_encode_raises(self, sample_image):
        jxl = pylibjxl.JXL()
        jxl.close()
        with pytest.raises(RuntimeError, match="closed"):
            jxl.encode(sample_image)

    def test_closed_decode_raises(self):
        jxl = pylibjxl.JXL()
        jxl.close()
        with pytest.raises(RuntimeError, match="closed"):
            jxl.decode(b"dummy")

    def test_close_idempotent(self):
        jxl = pylibjxl.JXL()
        jxl.close()
        jxl.close()  # Should not raise
        assert jxl.closed

    def test_without_context_manager(self, sample_image):
        jxl = pylibjxl.JXL(effort=4)
        data = jxl.encode(sample_image, lossless=True)
        result = jxl.decode(data)
        assert result.shape[2] == 3
        jxl.close()


class TestAsyncJXL:
    @pytest.mark.asyncio
    async def test_async_encode_decode_roundtrip(self, sample_image):
        img = sample_image
        async with pylibjxl.AsyncJXL(effort=4) as jxl:
            data = await jxl.encode_async(img, lossless=True)
            result = await jxl.decode_async(data)
            np.testing.assert_array_equal(result, img)
        assert jxl.closed

    @pytest.mark.asyncio
    async def test_async_multiple_operations(self, sample_image):
        img1 = sample_image
        img2 = np.ascontiguousarray(np.flipud(sample_image))
        
        async with pylibjxl.AsyncJXL(effort=4) as jxl:
            d1 = await jxl.encode_async(img1, lossless=True)
            d2 = await jxl.encode_async(img2, lossless=True)
            r1 = await jxl.decode_async(d1)
            r2 = await jxl.decode_async(d2)
            np.testing.assert_array_equal(r1, img1)
            np.testing.assert_array_equal(r2, img2)
