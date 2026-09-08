import numpy as np
import pytest

import pylibjxl


def test_probe_raw_buffer():
    # Create an image, encode to JXL, and test probe
    img = np.zeros((128, 256, 3), dtype=np.uint8)
    img[..., 0] = 255
    data = pylibjxl.encode(img, effort=1)

    info = pylibjxl.probe(data)
    assert info["width"] == 256
    assert info["height"] == 128
    assert info["channels"] == 3
    assert info["color_channels"] == 3
    assert not info["has_alpha"]
    assert info["bits_per_sample"] == 8
    assert not info["have_animation"]
    assert "suggested_threads" in info
    assert info["suggested_threads"] >= 1


def test_probe_rgba_buffer():
    img = np.zeros((64, 64, 4), dtype=np.uint8)
    img[..., 3] = 128
    data = pylibjxl.encode(img, effort=1)

    info = pylibjxl.probe(data)
    assert info["width"] == 64
    assert info["height"] == 64
    assert info["channels"] == 4
    assert info["color_channels"] == 3
    assert info["has_alpha"]


def test_probe_file_and_mmap(tmp_path):
    img = np.zeros((100, 150, 3), dtype=np.uint8)
    p = tmp_path / "test.jxl"
    pylibjxl.write(p, img, effort=1)

    info1 = pylibjxl.probe_file(p, use_mmap=False)
    assert info1["width"] == 150
    assert info1["height"] == 100

    info2 = pylibjxl.probe_file(p, use_mmap=True)
    assert info2["width"] == 150
    assert info2["height"] == 100


def test_context_probe():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    with pylibjxl.JXL(effort=1) as jxl:
        data = jxl.encode(img)
        info = jxl.probe(data)
        assert info["width"] == 50
        assert info["height"] == 50


@pytest.mark.asyncio
async def test_async_probe(tmp_path):
    img = np.zeros((80, 120, 3), dtype=np.uint8)
    data = await pylibjxl.encode_async(img, effort=1)
    info = await pylibjxl.probe_async(data)
    assert info["width"] == 120
    assert info["height"] == 80

    p = tmp_path / "async_probe.jxl"
    await pylibjxl.write_async(p, img, effort=1)
    info_file = await pylibjxl.probe_file_async(p)
    assert info_file["width"] == 120
    assert info_file["height"] == 80

    async with pylibjxl.AsyncJXL(effort=1) as ajxl:
        info_ctx = await ajxl.probe_async(data)
        assert info_ctx["width"] == 120
        assert info_ctx["height"] == 80

        info_file_ctx = await ajxl.probe_file_async(p)
        assert info_file_ctx["width"] == 120
        assert info_file_ctx["height"] == 80
