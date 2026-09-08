import io

import numpy as np
from PIL import Image

import pylibjxl

pylibjxl.register_pillow(override=True)


def test_pillow_registration():
    assert "JXL" in Image.ID
    assert ".jxl" in Image.EXTENSION
    assert Image.EXTENSION[".jxl"] == "JXL"
    assert Image.MIME["JXL"] == "image/jxl"


def test_pillow_open_and_load(tmp_path):
    # Create test image and write to JXL file
    arr = np.zeros((100, 150, 3), dtype=np.uint8)
    arr[:, :, 0] = 200  # Red
    arr[:, :, 1] = 100  # Green
    arr[:, :, 2] = 50  # Blue

    path = tmp_path / "test_pillow.jxl"
    pylibjxl.write(path, arr, effort=1)

    # Open with Pillow
    im = Image.open(path)
    assert im.format == "JXL"
    assert im.size == (150, 100)
    assert im.mode == "RGB"

    # Verify lazy loading: tile descriptor is present before load
    assert len(im.tile) == 1

    # Load pixel data
    im.load()
    assert len(im.tile) == 0
    arr_out = np.array(im)
    assert arr_out.shape == (100, 150, 3)
    # Near lossless comparison
    assert np.allclose(arr, arr_out, atol=15)


def test_pillow_save_rgb(tmp_path):
    arr = np.zeros((80, 80, 3), dtype=np.uint8)
    arr[20:60, 20:60] = 255
    im = Image.fromarray(arr)

    out_path = tmp_path / "saved_rgb.jxl"
    im.save(out_path, effort=1, quality=90)
    assert out_path.exists()

    # Re-read and check
    im2 = Image.open(out_path)
    assert im2.size == (80, 80)
    assert im2.mode == "RGB"


def test_pillow_save_rgba(tmp_path):
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[:, :, 3] = 180  # Alpha
    im = Image.fromarray(arr, mode="RGBA")

    out_path = tmp_path / "saved_rgba.jxl"
    im.save(out_path, effort=1, lossless=True)
    assert out_path.exists()

    im2 = Image.open(out_path)
    assert im2.size == (64, 64)
    assert im2.mode == "RGBA"
    arr_out = np.array(im2)
    assert np.array_equal(arr, arr_out)


def test_pillow_save_grayscale(tmp_path):
    arr = np.full((50, 75), 128, dtype=np.uint8)
    im = Image.fromarray(arr, mode="L")

    out_path = tmp_path / "saved_gray.jxl"
    im.save(out_path, effort=1, lossless=True)

    im2 = Image.open(out_path)
    assert im2.size == (75, 50)
    assert im2.mode == "L"
    arr_out = np.array(im2)
    assert np.array_equal(arr, arr_out)


def test_pillow_bytesio_stream():
    arr = np.zeros((32, 48, 3), dtype=np.uint8)
    im = Image.fromarray(arr)

    buf = io.BytesIO()
    im.save(buf, format="JXL", effort=1)
    buf.seek(0)

    im2 = Image.open(buf)
    assert im2.format == "JXL"
    assert im2.size == (48, 32)
    im2.load()
    assert np.array(im2).shape == (32, 48, 3)


def test_pillow_metadata_exif(tmp_path):
    arr = np.zeros((30, 30, 3), dtype=np.uint8)
    exif_bytes = b"Exif\x00\x00MM\x00*\x00\x00\x00\x08\x00\x00\x00\x00"

    path = tmp_path / "meta_exif.jxl"
    pylibjxl.write(path, arr, exif=exif_bytes, effort=1)

    im = Image.open(path)
    im.load()
    assert "exif" in im.info
    assert im.info["exif"] == exif_bytes


def test_pillow_cmyk_auto_convert(tmp_path):
    im_cmyk = Image.new("CMYK", (40, 40), color=(100, 50, 0, 0))
    out_path = tmp_path / "cmyk_converted.jxl"
    im_cmyk.save(out_path, effort=1)

    im_read = Image.open(out_path)
    assert im_read.mode == "RGB"
    assert im_read.size == (40, 40)
