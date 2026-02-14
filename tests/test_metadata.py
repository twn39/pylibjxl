"""Tests for EXIF / XMP / JUMBF metadata encoding and decoding."""

import numpy as np
import pytest

import pylibjxl

# ─── Descriptive metadata payloads ──────────────────────────────────────────────

# EXIF containing: Make=GeminiCamera, Model=CLI-v1, Software=pylibjxl-test, FNumber=2.8
EXIF_PAYLOAD = bytes.fromhex(
    "4578696600004d4d002a000000080004010f00020000000d0000003e0110000200000007"
    "0000004c013100020000000e00000054829d000c00000001000000620000000047656d"
    "696e6943616d6572610000434c492d7631000070796c69626a786c2d74657374004006"
    "666666666666"
)

# Descriptive XMP
XMP_PAYLOAD = (
    b'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
    b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b'<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
    b"<dc:description>Testing pylibjxl metadata preservation</dc:description>"
    b"<dc:creator>Gemini CLI</dc:creator>"
    b'</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
)

# Synthetic JUMBF payload
JUMBF_PAYLOAD = b"\x00\x00\x00\x1fjumb\x00\x00\x00\x11jumd\x00\x11\x00\x10"


# ─── Free Function: encode / decode roundtrip ───────────────────────────────────


class TestMetadataRoundtrip:
    """Test that metadata survives encode → decode roundtrip."""

    def test_exif_roundtrip(self, sample_image):
        img = sample_image
        data = pylibjxl.encode(img, exif=EXIF_PAYLOAD)
        result, meta = pylibjxl.decode(data, metadata=True)
        assert result.shape == img.shape
        assert "exif" in meta
        assert meta["exif"] == EXIF_PAYLOAD

    def test_xmp_roundtrip(self, sample_image):
        img = sample_image
        data = pylibjxl.encode(img, xmp=XMP_PAYLOAD)
        result, meta = pylibjxl.decode(data, metadata=True)
        assert result.shape == img.shape
        assert "xmp" in meta
        assert meta["xmp"] == XMP_PAYLOAD

    def test_jumbf_roundtrip(self, sample_image):
        img = sample_image
        data = pylibjxl.encode(img, jumbf=JUMBF_PAYLOAD)
        result, meta = pylibjxl.decode(data, metadata=True)
        assert result.shape == img.shape
        assert "jumbf" in meta
        assert meta["jumbf"] == JUMBF_PAYLOAD

    def test_multiple_metadata_types(self, sample_image):
        img = sample_image
        data = pylibjxl.encode(img, exif=EXIF_PAYLOAD, xmp=XMP_PAYLOAD)
        result, meta = pylibjxl.decode(data, metadata=True)
        assert result.shape == img.shape
        assert meta["exif"] == EXIF_PAYLOAD
        assert meta["xmp"] == XMP_PAYLOAD

    def test_all_three_metadata_types(self, sample_image):
        img = sample_image
        data = pylibjxl.encode(
            img, exif=EXIF_PAYLOAD, xmp=XMP_PAYLOAD, jumbf=JUMBF_PAYLOAD
        )
        result, meta = pylibjxl.decode(data, metadata=True)
        assert result.shape == img.shape
        assert len(meta) == 3
        assert meta["exif"] == EXIF_PAYLOAD
        assert meta["xmp"] == XMP_PAYLOAD
        assert meta["jumbf"] == JUMBF_PAYLOAD


# ─── Backward Compatibility ─────────────────────────────────────────────────────


class TestBackwardCompat:
    """decode() without metadata=True returns just ndarray (no regression)."""

    def test_decode_returns_array_by_default(self, sample_image):
        img = sample_image
        data = pylibjxl.encode(img)
        result = pylibjxl.decode(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == img.shape

    def test_decode_with_metadata_false(self, sample_image):
        img = sample_image
        data = pylibjxl.encode(img, exif=EXIF_PAYLOAD)
        result = pylibjxl.decode(data, metadata=False)
        assert isinstance(result, np.ndarray)

    def test_no_metadata_returns_empty_dict(self, sample_image):
        img = sample_image
        data = pylibjxl.encode(img)  # no metadata
        result, meta = pylibjxl.decode(data, metadata=True)
        assert isinstance(result, np.ndarray)
        assert isinstance(meta, dict)
        assert len(meta) == 0


# ─── Context Manager with Metadata ──────────────────────────────────────────────


class TestContextManagerMetadata:
    """Context manager encode/decode with metadata."""

    def test_context_encode_with_metadata(self, sample_image):
        img = sample_image
        with pylibjxl.JXL() as jxl:
            data = jxl.encode(img, exif=EXIF_PAYLOAD, xmp=XMP_PAYLOAD)
            result, meta = jxl.decode(data, metadata=True)
        assert meta["exif"] == EXIF_PAYLOAD
        assert meta["xmp"] == XMP_PAYLOAD

    def test_context_decode_without_metadata(self, sample_image):
        img = sample_image
        with pylibjxl.JXL() as jxl:
            data = jxl.encode(img, xmp=XMP_PAYLOAD)
            result = jxl.decode(data)
        assert isinstance(result, np.ndarray)


# ─── File I/O with Metadata ─────────────────────────────────────────────────────


class TestFileIOMetadata:
    """File read/write with metadata."""

    def test_write_read_with_exif(self, tmp_path, sample_image):
        img = sample_image
        path = tmp_path / "meta.jxl"
        pylibjxl.write(path, img, exif=EXIF_PAYLOAD)
        result, meta = pylibjxl.read(path, metadata=True)
        assert result.shape == img.shape
        assert meta["exif"] == EXIF_PAYLOAD

    def test_write_read_no_metadata(self, tmp_path, sample_image):
        img = sample_image
        path = tmp_path / "plain.jxl"
        pylibjxl.write(path, img)
        result = pylibjxl.read(path)
        assert isinstance(result, np.ndarray)

    def test_context_write_read_metadata(self, tmp_path, sample_image):
        img = sample_image
        path = tmp_path / "ctx_meta.jxl"
        with pylibjxl.JXL() as jxl:
            jxl.write(path, img, xmp=XMP_PAYLOAD)
            result, meta = jxl.read(path, metadata=True)
        assert meta["xmp"] == XMP_PAYLOAD


# ─── Async ──────────────────────────────────────────────────────────────────────


class TestAsyncMetadata:
    """Async encode/decode/read/write with metadata."""

    @pytest.mark.asyncio
    async def test_async_encode_decode_metadata(self, sample_image):
        img = sample_image
        data = await pylibjxl.encode_async(img, exif=EXIF_PAYLOAD)
        result, meta = await pylibjxl.decode_async(data, metadata=True)
        assert meta["exif"] == EXIF_PAYLOAD

    @pytest.mark.asyncio
    async def test_async_file_roundtrip(self, tmp_path, sample_image):
        img = sample_image
        path = tmp_path / "async_meta.jxl"
        await pylibjxl.write_async(path, img, exif=EXIF_PAYLOAD, xmp=XMP_PAYLOAD)
        result, meta = await pylibjxl.read_async(path, metadata=True)
        assert meta["exif"] == EXIF_PAYLOAD
        assert meta["xmp"] == XMP_PAYLOAD

    @pytest.mark.asyncio
    async def test_async_context_metadata(self, sample_image):
        img = sample_image
        async with pylibjxl.AsyncJXL() as jxl:
            data = await jxl.encode_async(img, xmp=XMP_PAYLOAD)
            result, meta = await jxl.decode_async(data, metadata=True)
        assert meta["xmp"] == XMP_PAYLOAD
