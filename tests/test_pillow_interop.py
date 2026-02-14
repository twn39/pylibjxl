import io

import numpy as np
from PIL import Image

import pylibjxl


def test_pillow_jpeg_decode_interop(real_image_bytes):
    """Ensure pylibjxl decodes JPEGs the same way (or very similarly) as Pillow."""
    # Decode with pylibjxl
    img_jxl = pylibjxl.decode_jpeg(real_image_bytes)

    # Decode with Pillow
    img_pil_raw = Image.open(io.BytesIO(real_image_bytes))
    img_pil = np.array(img_pil_raw.convert("RGB"))

    assert img_jxl.shape == img_pil.shape

    # libjpeg-turbo (used by pylibjxl) and Pillow (also often uses libjpeg-turbo)
    # might have slight differences depending on versions and flags,
    # but they should be extremely close.
    # We use a small tolerance because of potential differences in colorspace conversion implementations.
    mean_diff = np.mean(np.abs(img_jxl.astype(float) - img_pil.astype(float)))
    assert mean_diff < 1.0


def test_pillow_jpeg_encode_interop(sample_image):
    """Ensure JPEGs encoded by pylibjxl can be read by Pillow."""
    quality = 90

    # Encode with pylibjxl
    jpeg_data = pylibjxl.encode_jpeg(sample_image, quality=quality)

    # Decode with Pillow
    img_pil_raw = Image.open(io.BytesIO(jpeg_data))
    img_pil = np.array(img_pil_raw.convert("RGB"))

    assert img_pil.shape == sample_image.shape

    # Re-decode with pylibjxl to check consistency
    img_jxl = pylibjxl.decode_jpeg(jpeg_data)

    # pylibjxl decode vs Pillow decode of the same pylibjxl-encoded buffer
    # Even with the same buffer, different decoders or flags (like TJFLAG_FASTDCT)
    # might produce slightly different output.
    mean_diff = np.mean(np.abs(img_jxl.astype(float) - img_pil.astype(float)))
    assert mean_diff < 1.0


def test_pillow_to_jxl_roundtrip(sample_image):
    """Test workflow: Pillow Image -> pylibjxl JXL -> numpy -> Pillow."""
    # Start with Pillow
    pil_img = Image.fromarray(sample_image)

    # To numpy for pylibjxl
    np_img = np.array(pil_img)

    # Encode JXL
    jxl_data = pylibjxl.encode(np_img, lossless=True)

    # Decode JXL
    decoded_np = pylibjxl.decode(jxl_data)

    # Back to Pillow
    final_pil = Image.fromarray(decoded_np)

    # Check
    np.testing.assert_array_equal(np.array(final_pil), sample_image)


def test_pillow_rgba_to_jxl_lossless(sample_image_rgba):
    """Test RGBA roundtrip with Pillow."""
    pil_img = Image.fromarray(sample_image_rgba, mode="RGBA")

    # Encode
    jxl_data = pylibjxl.encode(np.array(pil_img), lossless=True)

    # Decode
    decoded_np = pylibjxl.decode(jxl_data)

    # Check
    assert decoded_np.shape[2] == 4
    np.testing.assert_array_equal(decoded_np, sample_image_rgba)


def test_jpeg_to_jxl_transcode_pillow_verify(real_image_bytes):
    """Transcode JPEG to JXL and verify the result with Pillow."""
    # Transcode
    jxl_data = pylibjxl.jpeg_to_jxl(real_image_bytes)

    # Decode JXL to pixels
    img_from_jxl = pylibjxl.decode(jxl_data)

    # Original JPEG decoded with Pillow
    img_pil = np.array(Image.open(io.BytesIO(real_image_bytes)).convert("RGB"))

    # Should be very similar to the original JPEG pixels
    # (Note: JPEG to JXL transcoding is lossless for the JPEG stream,
    # but the decoded pixels should match the original JPEG decoded pixels).
    mean_diff = np.mean(np.abs(img_from_jxl.astype(float) - img_pil.astype(float)))
    assert mean_diff < 1.0


def test_pillow_jpeg_metadata_interop(sample_image):
    """Ensure EXIF metadata survives Pillow -> pylibjxl -> Pillow roundtrip via JPEG."""
    # Valid EXIF containing: Make=GeminiCamera, Model=CLI-v1, Software=pylibjxl-test
    exif_payload = bytes.fromhex(
        "4578696600004d4d002a000000080004010f00020000000d0000003e0110000200000007"
        "0000004c013100020000000e00000054829d000c00000001000000620000000047656d"
        "696e6943616d6572610000434c492d7631000070796c69626a786c2d74657374004006"
        "666666666666"
    )

    # Encode JPEG with pylibjxl (Note: currently pylibjxl.encode_jpeg doesn't support exif param,
    # but we can test if it's preserved in JXL transcoding)

    # Let's test JPEG -> JXL transcoding with metadata preservation
    # Create a JPEG with Pillow that has EXIF
    pil_img = Image.fromarray(sample_image)
    buf = io.BytesIO()
    # Pillow's way of adding EXIF is via 'exif' kwarg or info dict
    pil_img.save(buf, format="JPEG", exif=exif_payload)
    jpeg_with_exif = buf.getvalue()

    # Transcode to JXL with pylibjxl
    jxl_data = pylibjxl.jpeg_to_jxl(jpeg_with_exif)

    # Reconstruct JPEG with pylibjxl
    reconstructed_jpeg = pylibjxl.jxl_to_jpeg(jxl_data)

    # Read back with Pillow
    Image.open(io.BytesIO(reconstructed_jpeg))

    # Check EXIF (Pillow's getexif() might return an object, we check raw if possible)
    # The simplest is to check if it's there.
    # JXL transcoding preserves the whole JPEG stream, so it should be exact.
    assert reconstructed_jpeg == jpeg_with_exif


def test_pillow_jpeg_transcode_io_interop(tmp_path, sample_image):
    """Verify that file conversions by pylibjxl produce valid JPEGs for Pillow."""
    jpg_path = tmp_path / "test.jpg"
    jxl_path = tmp_path / "test.jxl"
    rec_jpg_path = tmp_path / "rec.jpg"

    # Write JPEG with Pillow
    pil_img = Image.fromarray(sample_image)
    pil_img.save(jpg_path, quality=90)

    # Convert JPEG -> JXL with pylibjxl
    pylibjxl.convert_jpeg_to_jxl(jpg_path, jxl_path)

    # Convert JXL -> JPEG with pylibjxl
    pylibjxl.convert_jxl_to_jpeg(jxl_path, rec_jpg_path)

    # Open reconstructed JPEG with Pillow
    pil_img_rec = Image.open(rec_jpg_path)
    assert np.array(pil_img_rec).shape == sample_image.shape

    # Should be identical to the Pillow-generated JPEG
    assert rec_jpg_path.read_bytes() == jpg_path.read_bytes()


def test_pillow_metadata_parsing_interop(sample_image):
    """Verify that metadata extracted by pylibjxl is valid and parsable by Pillow."""
    import xml.etree.ElementTree as ET

    from PIL import Image

    # 1. Prepare descriptive metadata
    exif_payload = bytes.fromhex(
        "4578696600004d4d002a000000080004010f00020000000d0000003e0110000200000007"
        "0000004c013100020000000e00000054829d000c00000001000000620000000047656d"
        "696e6943616d6572610000434c492d7631000070796c69626a786c2d74657374004006"
        "666666666666"
    )
    xmp_payload = (
        b'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        b'<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
        b"<dc:description>Cross-validation test</dc:description>"
        b'</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
    )

    # 2. Encode with pylibjxl
    jxl_data = pylibjxl.encode(sample_image, exif=exif_payload, xmp=xmp_payload)

    # 3. Decode with pylibjxl
    _, meta = pylibjxl.decode(jxl_data, metadata=True)

    # 4. Cross-validate EXIF with Pillow
    # Wrap in a dummy JPEG to let Pillow parse it (or use Image.Exif)
    # Simpler: Create a JPEG with this EXIF and read it back.
    buf = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buf, format="JPEG", exif=meta["exif"])
    parsed_exif = Image.open(buf).getexif()

    # Check tags
    # 0x010f = Make, 0x0110 = Model
    assert parsed_exif[0x010F] == "GeminiCamera"
    assert parsed_exif[0x0110] == "CLI-v1"
    assert parsed_exif[0x0131] == "pylibjxl-test"

    # 5. Cross-validate XMP with ElementTree
    # XMP is just XML, but might have xpacket headers.
    # We find the <x:xmpmeta> start and end to parse the core XML.
    xmp_bytes = meta["xmp"]
    start_tag = b"<x:xmpmeta"
    end_tag = b"</x:xmpmeta>"
    start_idx = xmp_bytes.find(start_tag)
    end_idx = xmp_bytes.find(end_tag) + len(end_tag)

    if start_idx != -1 and end_idx != -1:
        core_xml = xmp_bytes[start_idx:end_idx]
        root = ET.fromstring(core_xml)
        # Find the description (namespaced)
        desc = root.find(".//{http://purl.org/dc/elements/1.1/}description")
        assert desc is not None
        assert desc.text == "Cross-validation test"
    else:
        # Fallback for simple exact match if headers weren't found/needed
        assert b"Cross-validation test" in xmp_bytes
