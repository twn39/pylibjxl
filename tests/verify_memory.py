import asyncio
import gc
import os
import psutil
import time
import numpy as np
from pylibjxl import AsyncJXL

def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

async def stress_test(iterations=100, concurrency=10, threads=4):
    print(f"Starting stress test: {iterations} iterations, {concurrency} concurrent tasks, {threads} runner threads")
    
    # Load real image
    from pylibjxl import decode_jpeg
    img_path = "images/test.jpg"
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    img = decode_jpeg(img_bytes)
    print(f"Loaded image {img_path}: {img.shape}")
    
    # Initial memory
    gc.collect()
    start_mem = get_process_memory_mb()
    print(f"Initial Memory: {start_mem:.2f} MB")
    
    async with AsyncJXL(threads=threads) as runner:
        try:
            for i in range(iterations):
                tasks = []
                for _ in range(concurrency):
                    tasks.append(runner.encode_async(img, effort=1)) # Use effort=1 for speed
                
                results = await asyncio.gather(*tasks)
                
                # Optional: Decode back to ensure full cycle
                decode_tasks = [runner.decode_async(data) for data in results]
                await asyncio.gather(*decode_tasks)
                
                if (i + 1) % 50 == 0:
                    gc.collect()
                    current_mem = get_process_memory_mb()
                    diff = current_mem - start_mem
                    print(f"Iteration {i+1}: {current_mem:.2f} MB (Delta: {diff:+.2f} MB)")
        except Exception as e:
            print(f"Error: {e}")
        
    gc.collect()
    end_mem = get_process_memory_mb()
    print(f"Final Memory: {end_mem:.2f} MB (Total Delta: {end_mem - start_mem:+.2f} MB)")

if __name__ == "__main__":
    asyncio.run(stress_test())
