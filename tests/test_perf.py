import asyncio
import pylibjxl
import numpy as np
import time
import pytest

@pytest.mark.asyncio
async def test_async_encode_decode():
    width, height = 100, 100
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    # Test async encode
    jxl_data = await pylibjxl.encode_async(img, effort=4)
    assert len(jxl_data) > 0
    
    # Test async decode
    decoded_img = await pylibjxl.decode_async(jxl_data)
    assert decoded_img.shape == img.shape

@pytest.mark.asyncio
async def test_concurrent_processing():
    width, height = 200, 200
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    # Measure serial time
    start = time.perf_counter()
    jxl1 = pylibjxl.encode(img, effort=7)
    jxl2 = pylibjxl.encode(img, effort=7)
    serial_time = time.perf_counter() - start
    
    # Measure concurrent time
    start = time.perf_counter()
    tasks = [
        pylibjxl.encode_async(img, effort=7),
        pylibjxl.encode_async(img, effort=7)
    ]
    await asyncio.gather(*tasks)
    concurrent_time = time.perf_counter() - start
    
    print(f"\nSerial time: {serial_time:.4f}s")
    print(f"Concurrent time: {concurrent_time:.4f}s")
    
    # Since we release the GIL, concurrent time should be significantly less than serial time
    # (assuming multiple cores are available)
    assert concurrent_time < serial_time * 0.9
