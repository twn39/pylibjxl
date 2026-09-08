"""Concurrency tests for RunnerPool — verifies true parallel execution.

Uses real test images from images/test.jpg via the ``sample_image`` session fixture.
Run with:
    uv run pytest tests/test_concurrency.py -v
    uv run pytest tests/test_concurrency.py -v --benchmark-only  # benchmarks only
"""

import asyncio
import time

import numpy as np
import pytest

import pylibjxl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _gather_encode(n, image, effort=3, use_context=False):
    """Run n concurrent encode operations, return (results, elapsed)."""
    if use_context:
        async with pylibjxl.AsyncJXL() as jxl:
            start = time.perf_counter()
            tasks = [jxl.encode_async(image, effort=effort) for _ in range(n)]
            results = await asyncio.gather(*tasks)
            return results, time.perf_counter() - start
    else:
        start = time.perf_counter()
        tasks = [pylibjxl.encode_async(image, effort=effort) for _ in range(n)]
        results = await asyncio.gather(*tasks)
        return results, time.perf_counter() - start


async def _gather_decode(n, jxl_data, use_context=False):
    """Run n concurrent decode operations, return (results, elapsed)."""
    if use_context:
        async with pylibjxl.AsyncJXL() as jxl:
            start = time.perf_counter()
            tasks = [jxl.decode_async(jxl_data) for _ in range(n)]
            results = await asyncio.gather(*tasks)
            return results, time.perf_counter() - start
    else:
        start = time.perf_counter()
        tasks = [pylibjxl.decode_async(jxl_data) for _ in range(n)]
        results = await asyncio.gather(*tasks)
        return results, time.perf_counter() - start


# ---------------------------------------------------------------------------
# Correctness Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_encode_correctness(sample_image):
    """All concurrent encodes produce identical output."""
    n = 8
    results, _ = await _gather_encode(n, sample_image)
    assert len(results) == n
    for r in results:
        assert r == results[0], "Concurrent encodes produced different output"


@pytest.mark.asyncio
async def test_concurrent_decode_correctness(sample_image):
    """All concurrent decodes produce identical output."""
    jxl_data = pylibjxl.encode(sample_image, effort=3)
    n = 8
    results, _ = await _gather_decode(n, jxl_data)
    assert len(results) == n
    for r in results:
        np.testing.assert_array_equal(r, results[0])


@pytest.mark.asyncio
async def test_concurrent_roundtrip(sample_image):
    """Concurrent encode → concurrent decode produces the same image."""
    n = 4
    encoded, _ = await _gather_encode(n, sample_image, effort=3)
    decoded, _ = await _gather_decode(n, encoded[0])
    for d in decoded:
        assert d.shape == sample_image.shape
        assert d.dtype == np.uint8


@pytest.mark.asyncio
async def test_context_manager_concurrent_encode(sample_image):
    """AsyncJXL context manager supports concurrent encode."""
    n = 8
    results, _ = await _gather_encode(n, sample_image, use_context=True)
    assert len(results) == n
    for r in results:
        assert r == results[0]


@pytest.mark.asyncio
async def test_context_manager_concurrent_decode(sample_image):
    """AsyncJXL context manager supports concurrent decode."""
    jxl_data = pylibjxl.encode(sample_image, effort=3)
    n = 8
    results, _ = await _gather_decode(n, jxl_data, use_context=True)
    assert len(results) == n
    for r in results:
        np.testing.assert_array_equal(r, results[0])


# ---------------------------------------------------------------------------
# Scaling Tests — verify throughput improves with concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_encode_scales_with_concurrency(sample_image):
    """Concurrent encoding should be faster than serial for 4 tasks."""
    serial_start = time.perf_counter()
    for _ in range(4):
        pylibjxl.encode(sample_image, effort=3)
    serial_time = time.perf_counter() - serial_start

    _, concurrent_time = await _gather_encode(4, sample_image, effort=3)

    speedup = serial_time / concurrent_time
    print(f"\n  Serial: {serial_time:.3f}s | Concurrent(4): {concurrent_time:.3f}s | Speedup: {speedup:.2f}x")
    # Should see at least some speedup on multi-core machines
    assert concurrent_time < serial_time * 1.5, (
        f"Concurrent ({concurrent_time:.3f}s) was not faster than serial ({serial_time:.3f}s)"
    )


@pytest.mark.asyncio
async def test_decode_scales_with_concurrency(sample_image):
    """Concurrent decoding should be faster than serial for 4 tasks."""
    jxl_data = pylibjxl.encode(sample_image, effort=3)

    serial_start = time.perf_counter()
    for _ in range(4):
        pylibjxl.decode(jxl_data)
    serial_time = time.perf_counter() - serial_start

    _, concurrent_time = await _gather_decode(4, jxl_data)

    speedup = serial_time / concurrent_time
    print(f"\n  Serial: {serial_time:.3f}s | Concurrent(4): {concurrent_time:.3f}s | Speedup: {speedup:.2f}x")
    assert concurrent_time < serial_time * 1.5, (
        f"Concurrent ({concurrent_time:.3f}s) was not faster than serial ({serial_time:.3f}s)"
    )


# ---------------------------------------------------------------------------
# Benchmark Tests (only run with --benchmark-only or --benchmark-enable)
# ---------------------------------------------------------------------------


CONCURRENCY_LEVELS = [1, 2, 4, 8]


@pytest.mark.parametrize("n", CONCURRENCY_LEVELS)
def test_benchmark_concurrent_encode_free(benchmark, sample_image, n):
    """Benchmark concurrent encode using free functions."""
    async def _run():
        results, _ = await _gather_encode(n, sample_image, effort=3)
        return results

    results = benchmark(lambda: asyncio.run(_run()))
    assert len(results) == n


@pytest.mark.parametrize("n", CONCURRENCY_LEVELS)
def test_benchmark_concurrent_encode_context(benchmark, sample_image, n):
    """Benchmark concurrent encode using AsyncJXL context manager."""
    async def _run():
        results, _ = await _gather_encode(n, sample_image, effort=3, use_context=True)
        return results

    results = benchmark(lambda: asyncio.run(_run()))
    assert len(results) == n


@pytest.mark.parametrize("n", CONCURRENCY_LEVELS)
def test_benchmark_concurrent_decode(benchmark, sample_image, n):
    """Benchmark concurrent decode using AsyncJXL."""
    jxl_data = pylibjxl.encode(sample_image, effort=3)

    async def _run():
        results, _ = await _gather_decode(n, jxl_data, use_context=True)
        return results

    results = benchmark(lambda: asyncio.run(_run()))
    assert len(results) == n
