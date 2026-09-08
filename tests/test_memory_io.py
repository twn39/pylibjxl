import mmap

import numpy as np
import pytest

import pylibjxl


class TestBufferProtocolInput:
    """Test that all decode and transcode APIs accept any Python buffer protocol object."""

    def test_decode_from_bytearray(self, sample_image):
        jxl_data = pylibjxl.encode(sample_image, effort=4, lossless=True)
        ba = bytearray(jxl_data)
        decoded = pylibjxl.decode(ba)
        np.testing.assert_array_equal(decoded, sample_image)

    def test_decode_from_memoryview(self, sample_image):
        jxl_data = pylibjxl.encode(sample_image, effort=4, lossless=True)
        mv = memoryview(jxl_data)
        decoded = pylibjxl.decode(mv)
        np.testing.assert_array_equal(decoded, sample_image)

    def test_decode_from_mmap(self, tmp_path, sample_image):
        path = tmp_path / "test_mmap.jxl"
        pylibjxl.write(path, sample_image, effort=4, lossless=True)
        with open(path, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                decoded = pylibjxl.decode(mm)
                np.testing.assert_array_equal(decoded, sample_image)

    def test_decode_jpeg_from_bytearray(self, sample_image):
        jpeg_data = pylibjxl.encode_jpeg(sample_image, quality=90)
        ba = bytearray(jpeg_data)
        decoded = pylibjxl.decode_jpeg(ba)
        assert decoded.shape == sample_image.shape
        assert decoded.dtype == np.uint8

    def test_decode_jpeg_from_memoryview(self, sample_image):
        jpeg_data = pylibjxl.encode_jpeg(sample_image, quality=90)
        mv = memoryview(jpeg_data)
        decoded = pylibjxl.decode_jpeg(mv)
        assert decoded.shape == sample_image.shape

    def test_transcode_from_memoryview(self, sample_image):
        jpeg_data = pylibjxl.encode_jpeg(sample_image, quality=90)
        mv = memoryview(jpeg_data)
        jxl_bytes = pylibjxl.jpeg_to_jxl(mv, effort=4)
        assert len(jxl_bytes) > 0
        restored_jpeg = pylibjxl.jxl_to_jpeg(memoryview(jxl_bytes))
        assert restored_jpeg == jpeg_data


class TestInPlaceDecoding:
    """Test in-place pre-allocated buffer decoding (out parameter)."""

    def test_jxl_decode_in_place(self, sample_image):
        jxl_data = pylibjxl.encode(sample_image, effort=4, lossless=True)
        out = np.zeros_like(sample_image)
        ret = pylibjxl.decode(jxl_data, out=out)
        assert ret is out
        np.testing.assert_array_equal(out, sample_image)

    def test_jxl_decode_in_place_with_metadata(self, sample_image):
        exif = b"ExifHeaderTest"
        jxl_data = pylibjxl.encode(sample_image, effort=4, lossless=True, exif=exif)
        out = np.zeros_like(sample_image)
        arr, meta = pylibjxl.decode(jxl_data, metadata=True, out=out)
        assert arr is out
        assert meta.get("exif") == exif
        np.testing.assert_array_equal(out, sample_image)

    def test_jxl_decode_in_place_shape_mismatch_raises(self, sample_image):
        jxl_data = pylibjxl.encode(sample_image, effort=4)
        wrong_out = np.zeros(
            (sample_image.shape[0] + 10, sample_image.shape[1], 3), dtype=np.uint8
        )
        with pytest.raises(ValueError, match="does not match"):
            pylibjxl.decode(jxl_data, out=wrong_out)

    def test_jpeg_decode_in_place(self, sample_image):
        jpeg_data = pylibjxl.encode_jpeg(sample_image, quality=90)
        out = np.zeros_like(sample_image)
        ret = pylibjxl.decode_jpeg(jpeg_data, out=out)
        assert ret is out
        assert out.shape == sample_image.shape

    def test_jpeg_decode_in_place_shape_mismatch_raises(self, sample_image):
        jpeg_data = pylibjxl.encode_jpeg(sample_image, quality=90)
        wrong_out = np.zeros(
            (sample_image.shape[0], sample_image.shape[1] + 10, 3), dtype=np.uint8
        )
        with pytest.raises(ValueError, match="does not match"):
            pylibjxl.decode_jpeg(jpeg_data, out=wrong_out)


class TestMmapAndFileIO:
    """Test mmap file reading and direct file transcoding."""

    def test_read_with_mmap(self, tmp_path, sample_image):
        path = tmp_path / "mmap_test.jxl"
        pylibjxl.write(path, sample_image, effort=4, lossless=True)
        img = pylibjxl.read(path, use_mmap=True)
        np.testing.assert_array_equal(img, sample_image)

    def test_read_with_mmap_and_out(self, tmp_path, sample_image):
        path = tmp_path / "mmap_out.jxl"
        pylibjxl.write(path, sample_image, effort=4, lossless=True)
        out = np.zeros_like(sample_image)
        ret = pylibjxl.read(path, out=out, use_mmap=True)
        assert ret is out
        np.testing.assert_array_equal(out, sample_image)

    def test_read_jpeg_with_mmap(self, tmp_path, sample_image):
        path = tmp_path / "mmap_test.jpg"
        pylibjxl.write_jpeg(path, sample_image, quality=90)
        img = pylibjxl.read_jpeg(path, use_mmap=True)
        assert img.shape == sample_image.shape

    def test_direct_file_transcode_roundtrip(self, tmp_path, sample_image):
        jpg_path = tmp_path / "orig.jpg"
        jxl_path = tmp_path / "transcoded.jxl"
        restored_jpg_path = tmp_path / "restored.jpg"

        pylibjxl.write_jpeg(jpg_path, sample_image, quality=92)
        orig_bytes = jpg_path.read_bytes()

        # Direct C++ file-to-file conversion
        pylibjxl.convert_jpeg_to_jxl(jpg_path, jxl_path, effort=4)
        assert jxl_path.exists()
        assert jxl_path.stat().st_size > 0

        pylibjxl.convert_jxl_to_jpeg(jxl_path, restored_jpg_path)
        assert restored_jpg_path.exists()
        restored_bytes = restored_jpg_path.read_bytes()

        # Lossless reconstruction must be bit-for-bit identical
        assert restored_bytes == orig_bytes


class TestContiguityDefense:
    """Test non-contiguous numpy arrays are safely handled."""

    def test_encode_non_contiguous(self, sample_image):
        # Transpose/slice creates non-contiguous array
        non_contig = sample_image[::-1, :, :]
        assert not non_contig.flags.c_contiguous
        data = pylibjxl.encode(non_contig, effort=4, lossless=True)
        decoded = pylibjxl.decode(data)
        np.testing.assert_array_equal(decoded, non_contig)

    def test_encode_jpeg_non_contiguous(self, sample_image):
        non_contig = sample_image[:, ::-1, :]
        assert not non_contig.flags.c_contiguous
        data = pylibjxl.encode_jpeg(non_contig, quality=90)
        decoded = pylibjxl.decode_jpeg(data)
        assert decoded.shape == non_contig.shape
