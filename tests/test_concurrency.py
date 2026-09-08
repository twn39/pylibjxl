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
    print(
        f"\n  Serial: {serial_time:.3f}s | Concurrent(4): {concurrent_time:.3f}s | Speedup: {speedup:.2f}x"
    )
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
    print(
        f"\n  Serial: {serial_time:.3f}s | Concurrent(4): {concurrent_time:.3f}s | Speedup: {speedup:.2f}x"
    )
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


# ---------------------------------------------------------------------------
# Concurrency, Pool Elasticity, and Timeout Tests
# ---------------------------------------------------------------------------


def test_codec_timeout_error_hierarchy():
    """CodecTimeoutError is a subclass of Python's built-in TimeoutError."""
    assert issubclass(pylibjxl.CodecTimeoutError, TimeoutError)
    assert issubclass(pylibjxl.CodecTimeoutError, Exception)


def test_pool_telemetry_and_elasticity(sample_image):
    """Verify dynamic on-demand expansion and runner pool telemetry."""
    import threading

    big_image = np.tile(sample_image, (2, 2, 1))
    with pylibjxl.JXL(pool_size=4, threads=1) as jxl:
        assert jxl.pool_size == 4
        assert jxl.threads_per_runner == 1
        # 1 eager warm-up runner created initially
        assert jxl.total_runners == 1
        assert jxl.available_runners == 1
        assert jxl.in_use_runners == 0

        # After single encode: runner returned to pool
        jxl.encode(sample_image, effort=1)
        assert jxl.total_runners == 1
        assert jxl.available_runners == 1
        assert jxl.in_use_runners == 0

        # Two concurrent operations trigger on-demand expansion
        barrier = threading.Barrier(2)

        def worker():
            barrier.wait()
            jxl.encode(big_image, effort=5)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert jxl.total_runners == 2
        assert jxl.available_runners == 2
        assert jxl.in_use_runners == 0


def test_sync_pool_timeout(sample_image):
    """With pool_size=1, a concurrent request with small timeout raises CodecTimeoutError."""
    import threading

    big_image = np.tile(sample_image, (2, 2, 1))
    with pylibjxl.JXL(pool_size=1) as jxl:
        errors = []

        def slow_worker():
            jxl.encode(big_image, effort=7)

        def timeout_worker():
            time.sleep(0.02)
            try:
                jxl.encode(big_image, effort=1, timeout=0.01)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=slow_worker)
        t2 = threading.Thread(target=timeout_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 1
        assert isinstance(errors[0], pylibjxl.CodecTimeoutError)
        assert isinstance(errors[0], TimeoutError)


@pytest.mark.asyncio
async def test_async_pool_timeout(sample_image):
    """AsyncJXL with pool_size=1 respects timeout and raises CodecTimeoutError."""
    big_image = np.tile(sample_image, (2, 2, 1))
    async with pylibjxl.AsyncJXL(pool_size=1) as jxl:
        task1 = asyncio.create_task(jxl.encode_async(big_image, effort=7))
        await asyncio.sleep(0.02)
        with pytest.raises(pylibjxl.CodecTimeoutError) as exc_info:
            await jxl.encode_async(big_image, effort=1, timeout=0.01)
        assert issubclass(exc_info.type, TimeoutError)
        await task1


@pytest.mark.asyncio
async def test_async_context_default_timeout(sample_image):
    """AsyncJXL uses context default timeout if per-call timeout is omitted."""
    big_image = np.tile(sample_image, (2, 2, 1))
    async with pylibjxl.AsyncJXL(pool_size=1, timeout=0.01) as jxl:
        task1 = asyncio.create_task(jxl.encode_async(big_image, effort=7))
        await asyncio.sleep(0.02)
        with pytest.raises(pylibjxl.CodecTimeoutError):
            await jxl.encode_async(big_image, effort=1)
        await task1
