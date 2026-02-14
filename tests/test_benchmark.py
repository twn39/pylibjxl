import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

try:
    import pillow_jxl
except ImportError:
    pillow_jxl = None

import pylibjxl


# Fixture to read the real image
@pytest.fixture(scope="module")
def real_image_bytes():
    image_path = Path("images/test.jpg")
    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")
    return image_path.read_bytes()


@pytest.fixture(scope="module")
def sample_image(real_image_bytes):
    """Decoded numpy array of the test image."""
    return pylibjxl.decode_jpeg(real_image_bytes)


@pytest.fixture(scope="module")
def sample_jpeg(real_image_bytes):
    """Raw JPEG bytes of the test image."""
    return real_image_bytes


@pytest.fixture(scope="module")
def sample_jxl(sample_image):
    """JXL bytes encoded from the test image."""
    return pylibjxl.encode(sample_image, effort=3)


def test_benchmark_jxl_encode(benchmark, sample_image):
    """Benchmark JXL encoding."""
    benchmark(pylibjxl.encode, sample_image, effort=3)


def test_benchmark_pillow_jxl_plugin_encode(benchmark, sample_image):
    """Benchmark JXL encoding with pillow-jxl-plugin."""
    if pillow_jxl is None:
        pytest.skip("pillow-jxl-plugin not installed")
    pil_img = Image.fromarray(sample_image)

    def _encode():
        buf = io.BytesIO()
        # effort=3 in pylibjxl is roughly speed=7 in pillow-jxl
        pil_img.save(buf, format="JXL", speed=7, distance=1.0)
        return buf.getvalue()

    benchmark(_encode)


def test_benchmark_jxl_decode(benchmark, sample_jxl):
    """Benchmark JXL decoding."""
    benchmark(pylibjxl.decode, sample_jxl)


def test_benchmark_pillow_jxl_plugin_decode(benchmark, sample_jxl):
    """Benchmark JXL decoding with pillow-jxl-plugin."""
    if pillow_jxl is None:
        pytest.skip("pillow-jxl-plugin not installed")

    def _decode():
        img = Image.open(io.BytesIO(sample_jxl))
        return np.array(img)

    benchmark(_decode)


def test_benchmark_jpeg_encode(benchmark, sample_image):
    """Benchmark JPEG encoding."""
    benchmark(pylibjxl.encode_jpeg, sample_image, quality=90)


def test_benchmark_pillow_jpeg_encode(benchmark, sample_image):
    """Benchmark JPEG encoding with Pillow."""
    pil_img = Image.fromarray(sample_image)

    def _encode():
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    benchmark(_encode)


def test_benchmark_jpeg_decode(benchmark, sample_jpeg):
    """Benchmark JPEG decoding."""
    benchmark(pylibjxl.decode_jpeg, sample_jpeg)


def test_benchmark_pillow_jpeg_decode(benchmark, sample_jpeg):
    """Benchmark JPEG decoding with Pillow."""

    def _decode():
        img = Image.open(io.BytesIO(sample_jpeg))
        return np.array(img)

    benchmark(_decode)


def test_benchmark_jpeg_to_jxl(benchmark, sample_jpeg):
    """Benchmark JPEG to JXL transcoding."""
    benchmark(pylibjxl.jpeg_to_jxl, sample_jpeg)


def test_benchmark_jxl_to_jpeg_reconstruction(benchmark, sample_jpeg):
    """Benchmark JXL to JPEG reconstruction (requires JXL from JPEG)."""
    # Create JXL from JPEG first (transcoding)
    jxl = pylibjxl.jpeg_to_jxl(sample_jpeg)

    benchmark(pylibjxl.jxl_to_jpeg, jxl)


@pytest.mark.parametrize("effort", [1, 4, 7])
def test_benchmark_jxl_encode_comparison(benchmark, sample_image, effort):
    """Compare pylibjxl vs pillow-jxl-plugin at same effort levels."""
    
    def _pylib():
        return pylibjxl.encode(sample_image, effort=effort)
    
    benchmark.extra_info['effort'] = effort
    benchmark(_pylib)


@pytest.mark.parametrize("effort", [1, 4, 7])
def test_benchmark_pillow_jxl_plugin_encode_comparison(benchmark, sample_image, effort):
    """Benchmark pillow-jxl-plugin at same effort levels."""
    if pillow_jxl is None:
        pytest.skip("pillow-jxl-plugin not installed")
    pil_img = Image.fromarray(sample_image)

    def _pillow():
        buf = io.BytesIO()
        # Correct parameter is 'effort', same as pylibjxl
        pil_img.save(buf, format="JXL", effort=effort)
        return buf.getvalue()

    benchmark.extra_info['effort'] = effort
    benchmark(_pillow)


# --- File I/O Benchmarks ---


def test_benchmark_jxl_write(benchmark, sample_image, tmp_path):
    """Benchmark writing JXL to file."""
    output_path = tmp_path / "bench.jxl"

    def _write():
        pylibjxl.write(output_path, sample_image, effort=3)

    benchmark(_write)
    # Cleanup happens automatically by tmp_path when test session ends
    # but we are overwriting in the loop, so disk usage is constant.


def test_benchmark_jxl_read(benchmark, sample_image, tmp_path):
    """Benchmark reading JXL from file."""
    input_path = tmp_path / "bench_read.jxl"
    pylibjxl.write(input_path, sample_image, effort=3)

    def _read():
        pylibjxl.read(input_path)

    benchmark(_read)


def test_benchmark_jpeg_write(benchmark, sample_image, tmp_path):
    """Benchmark writing JPEG to file."""
    output_path = tmp_path / "bench.jpg"

    def _write():
        pylibjxl.write_jpeg(output_path, sample_image, quality=90)

    benchmark(_write)


def test_benchmark_pillow_jpeg_write(benchmark, sample_image, tmp_path):
    """Benchmark writing JPEG to file with Pillow."""
    output_path = tmp_path / "bench_pillow.jpg"
    pil_img = Image.fromarray(sample_image)

    def _write():
        pil_img.save(output_path, quality=90)

    benchmark(_write)


def test_benchmark_jpeg_read(benchmark, sample_image, tmp_path):
    """Benchmark reading JPEG from file."""
    input_path = tmp_path / "bench_read.jpg"
    pylibjxl.write_jpeg(input_path, sample_image, quality=90)

    def _read():
        pylibjxl.read_jpeg(input_path)

    benchmark(_read)


def test_benchmark_pillow_jpeg_read(benchmark, sample_image, tmp_path):
    """Benchmark reading JPEG from file with Pillow."""
    input_path = tmp_path / "bench_read_pillow.jpg"
    pil_img = Image.fromarray(sample_image)
    pil_img.save(input_path, quality=90)

    def _read():
        img = Image.open(input_path)
        return np.array(img)

    benchmark(_read)
