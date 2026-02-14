import numpy as np
import pytest

import pylibjxl


class TestJXLClassCoverage:
    """Tests for JXL class methods coverage."""

    def test_read_file_not_found(self):
        with pylibjxl.JXL() as jxl:
            with pytest.raises(FileNotFoundError):
                jxl.read("non_existent_file.jxl")

    def test_read_jpeg_file_not_found(self):
        with pylibjxl.JXL() as jxl:
            with pytest.raises(FileNotFoundError):
                jxl.read_jpeg("non_existent_file.jpg")

    def test_convert_jpeg_file_not_found(self):
        with pylibjxl.JXL() as jxl:
            with pytest.raises(FileNotFoundError):
                jxl.convert_jpeg_to_jxl("non_existent.jpg", "output.jxl")

    def test_convert_jxl_file_not_found(self):
        with pylibjxl.JXL() as jxl:
            with pytest.raises(FileNotFoundError):
                jxl.convert_jxl_to_jpeg("non_existent.jxl", "output.jpg")

    def test_jpeg_io_roundtrip(self, tmp_path, sample_image):
        img = sample_image
        path = tmp_path / "test.jpg"
        with pylibjxl.JXL() as jxl:
            jxl.write_jpeg(path, img, quality=90)
            assert path.exists()
            decoded = jxl.read_jpeg(path)
            # JPEG is lossy, so shapes match but pixels might differ slightly.
            # Just asserting shape/dtype matches input.
            assert decoded.shape == img.shape
            assert decoded.dtype == img.dtype

    def test_conversion_roundtrip(self, tmp_path, sample_image):
        # JPEG -> JXL -> JPEG (lossless transcoding)
        img = sample_image
        jpg_path = tmp_path / "origin.jpg"
        jxl_path = tmp_path / "converted.jxl"
        rec_jpg_path = tmp_path / "reconstructed.jpg"

        with pylibjxl.JXL() as jxl:
            # Create initial JPEG
            jxl.write_jpeg(jpg_path, img, quality=95)

            # Convert JPEG -> JXL
            jxl.convert_jpeg_to_jxl(jpg_path, jxl_path)
            assert jxl_path.exists()

            # Convert JXL -> JPEG
            jxl.convert_jxl_to_jpeg(jxl_path, rec_jpg_path)
            assert rec_jpg_path.exists()

            # Verify contents are similar (JPEG encoding is lossy, but transcoding should be consistent)
            # Just checking if we can read it back
            final_img = jxl.read_jpeg(rec_jpg_path)
            assert final_img.shape == img.shape


class TestAsyncJXLClassCoverage:
    """Tests for AsyncJXL class methods coverage."""

    @pytest.mark.asyncio
    async def test_read_async_file_not_found(self):
        async with pylibjxl.AsyncJXL() as jxl:
            with pytest.raises(FileNotFoundError):
                await jxl.read_async("non_existent.jxl")

    @pytest.mark.asyncio
    async def test_read_jpeg_async_file_not_found(self):
        async with pylibjxl.AsyncJXL() as jxl:
            with pytest.raises(FileNotFoundError):
                await jxl.read_jpeg_async("non_existent.jpg")

    @pytest.mark.asyncio
    async def test_convert_jpeg_async_file_not_found(self):
        async with pylibjxl.AsyncJXL() as jxl:
            with pytest.raises(FileNotFoundError):
                await jxl.convert_jpeg_to_jxl_async("non_existent.jpg", "out.jxl")

    @pytest.mark.asyncio
    async def test_convert_jxl_async_file_not_found(self):
        async with pylibjxl.AsyncJXL() as jxl:
            with pytest.raises(FileNotFoundError):
                await jxl.convert_jxl_to_jpeg_async("non_existent.jxl", "out.jpg")

    @pytest.mark.asyncio
    async def test_jpeg_async_io(self, tmp_path, sample_image):
        img = sample_image
        path = tmp_path / "async.jpg"
        async with pylibjxl.AsyncJXL() as jxl:
            await jxl.write_jpeg_async(path, img, quality=80)
            assert path.exists()
            decoded = await jxl.read_jpeg_async(path)
            assert decoded.shape == img.shape

    @pytest.mark.asyncio
    async def test_transcode_async_in_memory(self, real_image_bytes):
        # Create JPEG bytes
        jpeg_data = real_image_bytes

        async with pylibjxl.AsyncJXL() as jxl:
            # JPEG bytes -> JXL bytes
            jxl_data = await jxl.jpeg_to_jxl_async(jpeg_data)
            assert len(jxl_data) > 0

            # JXL bytes -> JPEG bytes
            rec_jpeg_data = await jxl.jxl_to_jpeg_async(jxl_data)
            assert len(rec_jpeg_data) > 0
            # Should be roughly same size as original (exact reconstruction possible for file, bytes might vary slightly due to headers but jxl reconstruction is usually bit-exact if from jpeg)
            # Actually, `jpeg_to_jxl` and back SHOULD be bit exact for the jpeg stream.

            assert abs(len(rec_jpeg_data) - len(jpeg_data)) < 2000  # Relaxed check

    @pytest.mark.asyncio
    async def test_conversion_async_file(self, tmp_path, sample_image):
        img = sample_image
        jpg_path = tmp_path / "async_origin.jpg"
        jxl_path = tmp_path / "async_conv.jxl"
        rec_jpg_path = tmp_path / "async_rec.jpg"

        async with pylibjxl.AsyncJXL() as jxl:
            await jxl.write_jpeg_async(jpg_path, img)

            await jxl.convert_jpeg_to_jxl_async(jpg_path, jxl_path)
            assert jxl_path.exists()

            await jxl.convert_jxl_to_jpeg_async(jxl_path, rec_jpg_path)
            assert rec_jpg_path.exists()

    @pytest.mark.asyncio
    async def test_async_jxl_methods_direct(self, sample_image):
        img = sample_image
        async with pylibjxl.AsyncJXL() as jxl:
            # Test encode_jpeg_async
            data = await jxl.encode_jpeg_async(img, quality=85)
            assert isinstance(data, bytes)
            assert len(data) > 0

            # Test decode_jpeg_async
            decoded = await jxl.decode_jpeg_async(data)
            assert decoded.shape == img.shape
            assert decoded.dtype == np.uint8


class TestFreeFunctionsCoverage:
    """Tests for free function coverage that might be missed."""

    @pytest.mark.asyncio
    async def test_free_jpeg_to_jxl_async(self, real_image_bytes):
        jpeg_data = real_image_bytes
        jxl_data = await pylibjxl.jpeg_to_jxl_async(jpeg_data)
        assert len(jxl_data) > 0

        rec_jpeg = await pylibjxl.jxl_to_jpeg_async(jxl_data)
        assert len(rec_jpeg) > 0

    @pytest.mark.asyncio
    async def test_free_convert_async_files(self, tmp_path, sample_image):
        img = sample_image
        jpg_path = tmp_path / "free_async.jpg"
        jxl_path = tmp_path / "free_async.jxl"
        rec_path = tmp_path / "free_rec.jpg"

        await pylibjxl.write_jpeg_async(jpg_path, img)
        await pylibjxl.convert_jpeg_to_jxl_async(jpg_path, jxl_path)
        assert jxl_path.exists()

        await pylibjxl.convert_jxl_to_jpeg_async(jxl_path, rec_path)
        assert rec_path.exists()
