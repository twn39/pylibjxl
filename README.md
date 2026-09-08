<div align="center">

# pylibjxl

**Fast Python bindings for JPEG XL (libjxl) and JPEG (libjpeg-turbo)**

[![CI](https://github.com/twn39/pylibjxl/actions/workflows/build.yml/badge.svg)](https://github.com/twn39/pylibjxl/actions/workflows/build.yml)
[![PyPI version](https://img.shields.io/pypi/v/pylibjxl.svg)](https://pypi.org/project/pylibjxl/)
[![Python versions](https://img.shields.io/pypi/pyversions/pylibjxl.svg)](https://pypi.org/project/pylibjxl/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

---

`pylibjxl` provides efficient, high-performance Python bindings for [libjxl](https://github.com/libjxl/libjxl) and [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo). Built with [nanobind](https://github.com/wjakob/nanobind), it features **GIL-free** encoding/decoding and **native async** support for maximum throughput.

</div>

## ✨ Key Features

- 🚀 **High Performance** — C++ core releases the GIL during heavy computation for true multi-core scaling.
- 📦 **Metadata Excellence** — Full support for EXIF, XMP, and JUMBF metadata, plus ICC color profile management.
- ⚡ **Async-First & Backpressure** — Native `asyncio` integration with double-layer semaphore backpressure for web services.
- 🎯 **Elastic RunnerPool** — On-demand dynamic expansion, idle runner reaping, and timeout protection (`CodecTimeoutError`).
- 🖼️ **NumPy Native & Zero-Copy** — In-place decode (`out=array`) and Buffer Protocol support for zero memory allocation.
- 🔄 **Lossless JPEG Transcoding** — Bit-perfect JPEG ↔ JXL roundtrips without pixel decoding.

---

## 🛠️ Installation

### Install from PyPI
```bash
# Recommended: Using uv
uv pip install pylibjxl

# Or via standard pip
pip install pylibjxl
```

### Install from Source
```bash
uv pip install git+https://github.com/twn39/pylibjxl.git --recursive
```

---

## Quick Start

### 🖼️ Basic In-Memory Operations
```python
import numpy as np
import pylibjxl

# Create a test image (Height, Width, Channels)
image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

# Encode to JXL bytes
data = pylibjxl.encode(image, effort=7, distance=1.0)

# Decode back to NumPy array
decoded = pylibjxl.decode(data)
```

### 💾 File I/O & Metadata
`pylibjxl` handles EXIF and XMP metadata seamlessly.
```python
# Write an image with EXIF metadata
exif_data = b"Raw EXIF bytes..."
pylibjxl.write("output.jxl", image, effort=9, exif=exif_data)

# Read image and its metadata
img, meta = pylibjxl.read("output.jxl", metadata=True)
print(f"Loaded image shape: {img.shape}")
print(f"EXIF size: {len(meta.get('exif', b''))} bytes")
```

### 🔄 Lossless JPEG Transcoding
Reduce JPEG file size by ~20% without losing a single bit of information. The resulting `.jxl` can be restored to the exact original `.jpg`.
```python
# Convert JPEG to JXL losslessly
pylibjxl.convert_jpeg_to_jxl("input.jpg", "input.jxl")

# Restore the bit-identical original JPEG
pylibjxl.convert_jxl_to_jpeg("input.jxl", "restored.jpg")
```

### ⚡ Async Support
High-performance non-blocking I/O for web servers and data pipelines.
```python
import asyncio

async def main():
    # Async encoding
    data = await pylibjxl.encode_async(image, distance=0.0)
    
    # Async file reading
    img = await pylibjxl.read_async("input.jxl")

asyncio.run(main())
```

### 🏗️ Batch Processing (Context Manager)
Using the `JXL` context manager maintains a persistent thread pool, providing a significant speedup for batch operations.
```python
# High-performance batch conversion
with pylibjxl.JXL(effort=7) as jxl:
    for i in range(100):
        img = jxl.read(f"input_{i}.jxl")
        # Process and save as high-quality JPEG
        jxl.write_jpeg(f"output_{i}.jpg", img, quality=95)
```

### 🚀 High-Concurrency Web Servers (FastAPI Example)
`pylibjxl` is engineered for high-concurrency production web servers and microservices. By combining **double-layer backpressure** with an **elastic RunnerPool**, it protects servers from memory exhaustion and CPU thrashing under traffic spikes.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, status
import pylibjxl

# Allocate an isolated codec with a pool of up to 8 runners and a 3.0s timeout
codec = pylibjxl.AsyncJXL(pool_size=8, timeout=3.0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with codec:
        yield

app = FastAPI(title="Image Service", lifespan=lifespan)

@app.post("/encode")
async def encode_image(file: UploadFile):
    content = await file.read()
    try:
        # Step 1: Decode JPEG (GIL released)
        image = await codec.decode_jpeg_async(content)

        # Step 2: Encode to JXL with per-operation timeout (e.g., 2.0s)
        # If all 8 runners are busy and cannot be acquired within 2.0s,
        # it raises CodecTimeoutError rather than queueing infinitely.
        jxl_bytes = await codec.encode_async(image, effort=5, timeout=2.0)
        return jxl_bytes

    except pylibjxl.CodecTimeoutError:
        # Gracefully handle server saturation
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image encoder is at maximum capacity; please retry shortly.",
        )
```

---

## 🧠 Concurrency & Resource Management (Deep Dive)

### 1. The Concurrency Challenge in libjxl
In the official Google `libjxl` architecture, `JxlResizableParallelRunner` is **stateful and not thread-safe**. A single runner instance cannot be shared across multiple threads or concurrent requests simultaneously.

### 2. The Throughput Formula: $M \times N \le \text{Cores}$
To maximize hardware utilization without causing CPU cache thrashing or context-switch storms, `pylibjxl` organizes concurrency according to the rule:

$$\text{Active Runners } (M) \times \text{Threads Per Runner } (N) \le \text{Available CPU Cores}$$

`pylibjxl` provides optimal presets for different workloads:

| Workload Mode | Configuration | Characteristics | Best For |
|:---|:---|:---|:---|
| **High-Concurrency Web Server** *(Default)* | `pool_size = Cores`, `threads = 1` | Maximizes global QPS; requests execute in parallel without CPU starvation. | FastAPI, Django, Celery, Tornado |
| **Low-Latency Single Image** | `pool_size = 1`, `threads = Cores` | Uses all cores to encode/decode a single image as fast as possible. | CLI tools, batch scripts, offline pipelines |
| **Auto-Balanced** | `pool_size = 0`, `threads = 0` | Automatically balances $M$ and $N$ based on detected CPU cores. | General production applications |

### 3. Elastic RunnerPool Lifecycle
- **Zero Cold-Start Latency**: Pre-allocates 1 warm runner upon creation so the first request incurs zero setup overhead.
- **On-Demand Dynamic Expansion**: Spawns additional runners only when concurrent load demands it, up to `max_pool_size`.
- **Idle Reaping**: Tracks timestamp activity and automatically deallocates idle runners beyond the baseline after `idle_timeout` (default 30s), reclaiming system memory during traffic lulls.

### 4. Double-Layer Backpressure
In Python web frameworks, asynchronous task queues (e.g. `asyncio.to_thread`) are unbounded by default. Under a traffic surge, thousands of coroutines can flood the thread pool queue, causing out-of-memory crashes.

`AsyncJXL` solves this with **Double-Layer Backpressure**:
1. **Event Loop Layer (`asyncio.Semaphore`)**: Restricts the maximum number of concurrent tasks entering the thread pool. Excess tasks wait on the event loop with bounded `asyncio.wait_for` timeouts.
2. **C++ Native Layer (`std::condition_variable`)**: Bounded `acquire(timeout)` ensures OS threads never block indefinitely.

```python
# Check real-time pool metrics
print(f"Total runners: {codec.total_runners}")
print(f"Available runners: {codec.available_runners}")
print(f"In-use runners: {codec.in_use_runners}")
```

### 5. Timeout Protection & `CodecTimeoutError`
Both sync and async operations accept a `timeout` argument in seconds. When the runner pool cannot satisfy an acquisition request within the specified time, it raises `pylibjxl.CodecTimeoutError`.

`CodecTimeoutError` inherits from Python's standard `TimeoutError`, allowing catch-all handling:
```python
try:
    data = await codec.encode_async(image, timeout=1.0)
except pylibjxl.CodecTimeoutError:
    # Specific codec timeout
    pass
except TimeoutError:
    # Standard Python timeout handler catches it too
    pass
```

---

## 📈 Performance & Stability

`pylibjxl` is engineered for high-throughput production environments.

### 🚀 pylibjxl vs. pillow-jxl Comparison

*Tested on Apple Silicon (8-core), 1440x960 RGB image:*

| Test Scenario | pillow-jxl (`num_threads=-1`) | pylibjxl (Default `threads=1`) | pylibjxl (Multi-thread `threads=8`) | Comparison Result |
|:---|:---:|:---:|:---:|:---|
| **JXL Decode Latency** | 19.68 ms | 43.26 ms | **17.79 ms** | **pylibjxl is ~11% faster** |
| **JXL Encode (effort=7)** | 139.00 ms | 321.00 ms | **117.00 ms** | **pylibjxl is ~19% faster** |
| **8-Task Concurrent Throughput** | Heavy context thrashing | **129.0 OPS** | - | **pylibjxl scales linearly** |

> [!NOTE]
> **Why does pillow-jxl seem faster on single-image defaults?**  
> `pillow-jxl` defaults to `num_threads = -1`, forcing all CPU cores onto a single image. While beneficial for single-image CLI scripts, this creates severe CPU thread contention in concurrent web servers (e.g. 8 simultaneous requests launch $8 \times 8 = 64$ threads).  
> `pylibjxl` defaults to `threads=1` per runner to optimize overall throughput. When configured to use equivalent multi-threading (`pylibjxl.JXL(threads=8)`), **pylibjxl outperforms pillow-jxl across both encode and decode**.

### 📊 Multi-Core Concurrency Throughput Scaling
*Evaluated with `pytest-benchmark` on Apple Silicon (1440x960 RGB image):*

| Concurrent Tasks | Total Latency (Mean) | Total Throughput | Multi-Core Scaling |
|:---:|:---:|:---:|:---:|
| **1 Task** | 25.75 ms | 38.8 tasks/sec | 1.00x |
| **2 Tasks** | 29.04 ms | 68.9 tasks/sec | **1.77x** |
| **4 Tasks** | 33.51 ms | 119.4 tasks/sec | **3.07x** |
| **8 Tasks** | 62.02 ms | **129.0 tasks/sec** | **3.32x** |

### 🛠️ Key Architectural Advantages
- **GIL-Free Execution**: The C++ core releases Python's Global Interpreter Lock (GIL) during all heavy encoding and decoding tasks.
- **Zero Memory Leaks**: Verified over 500+ consecutive rounds with stable memory footprint.
- **Zero-Copy Buffer Protocol**: `decode(data, out=ndarray)` writes directly into pre-allocated NumPy memory with zero intermediate allocations.
- **Event-Loop Purity**: Array contiguity conversions (`np.ascontiguousarray`) execute inside worker threads to ensure **0ms blocking** on the main asyncio event loop.

---

## 📂 API Reference

### 🖼️ JXL In-Memory Operations

#### `encode(input, effort=7, distance=1.0, lossless=False, decoding_speed=0, *, exif=None, xmp=None, jumbf=None) -> bytes`
#### `async encode_async(...) -> bytes`
Encodes a NumPy array into JPEG XL format.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `ndarray` | *required* | uint8 array of shape `(H, W, 3)` or `(H, W, 4)` |
| `effort` | `int` | `7` | Speed/size tradeoff `[1-11]`. 1=fastest, 11=best compression. |
| `distance` | `float` | `1.0` | Perceptual quality `[0.0-25.0]`. 0.0=lossless, 1.0=visually lossless. |
| `lossless` | `bool` | `False` | If `True`, enables mathematical lossless mode. |
| `decoding_speed` | `int` | `0` | Decoding speed tier `[0-4]`. 0=default, 4=fastest decoding. |
| `exif` | `bytes` | `None` | Optional raw EXIF metadata. |
| `xmp` | `bytes` | `None` | Optional raw XMP (XML) metadata. |
| `jumbf` | `bytes` | `None` | Optional raw JUMBF metadata. |

```python
# Synchronous encoding
data = pylibjxl.encode(image, effort=9, lossless=True)

# Asynchronous encoding
data = await pylibjxl.encode_async(image, distance=0.5)
```

---

#### `decode(data, *, metadata=False) -> ndarray | tuple[ndarray, dict]`
#### `async decode_async(...) -> ndarray | tuple[ndarray, dict]`
Decodes JPEG XL bytes back into a NumPy array.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `bytes` | *required* | JPEG XL encoded bytes. |
| `metadata` | `bool` | `False` | If `True`, returns a tuple including a metadata dictionary. |

```python
# Basic decode
img = pylibjxl.decode(jxl_bytes)

# Decode with metadata
img, meta = await pylibjxl.decode_async(jxl_bytes, metadata=True)
print(f"EXIF size: {len(meta.get('exif', b''))} bytes")
```

---

### 💾 JXL File I/O

#### `read(path, *, metadata=False)` / `async read_async(...)`
Reads a `.jxl` file from disk and decodes it.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str | Path` | *required* | Path to the source `.jxl` file. |
| `metadata` | `bool` | `False` | Whether to return metadata alongside the image. |

```python
img = pylibjxl.read("input.jxl")
img, meta = await pylibjxl.read_async("input.jxl", metadata=True)
```

---

#### `write(path, image, ...)` / `async write_async(...)`
Encodes a NumPy array and writes it directly to a `.jxl` file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str | Path` | *required* | Destination file path. |
| `image` | `ndarray` | *required* | The image data to encode. |
| `...` | | | Supports all parameters from `encode()`. |

```python
pylibjxl.write("output.jxl", image, effort=7, distance=1.0)
await pylibjxl.write_async("output.jxl", image, lossless=True)
```

---

### 📷 JPEG Support (libjpeg-turbo)

#### `encode_jpeg(input, quality=95) -> bytes` / `async encode_jpeg_async(...)`
Encodes a NumPy array to JPEG bytes using high-speed libjpeg-turbo.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `ndarray` | *required* | uint8 array of shape `(H, W, 3)`. |
| `quality` | `int` | `95` | JPEG quality factor `[1-100]`. |

```python
jpeg_bytes = pylibjxl.encode_jpeg(image, quality=90)
```

---

#### `decode_jpeg(data) -> ndarray` / `async decode_jpeg_async(...)`
Decodes JPEG bytes to a NumPy RGB array.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `bytes` | *required* | JPEG encoded bytes. |

```python
image = pylibjxl.decode_jpeg(jpeg_bytes)
```

---

#### `read_jpeg(path)` / `write_jpeg(path, image, quality=95)`
Stand-alone JPEG file I/O operations using libjpeg-turbo.

```python
img = pylibjxl.read_jpeg("photo.jpg")
pylibjxl.write_jpeg("output.jpg", img, quality=85)
```

---

### 🔄 Lossless Transcoding (JPEG ↔ JXL)

#### `jpeg_to_jxl(data, effort=7) -> bytes` / `async jpeg_to_jxl_async(...)`
Transcodes raw JPEG bytes into a JPEG XL container losslessly.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `bytes` | *required* | Original JPEG bytes. |
| `effort` | `int` | `7` | Transcoding effort `[1-11]`. |

```python
jxl_data = pylibjxl.jpeg_to_jxl(jpeg_bytes)
```

---

#### `jxl_to_jpeg(data) -> bytes` / `async jxl_to_jpeg_async(...)`
Restores the original JPEG bytes from a transcoded JXL file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `bytes` | *required* | Transcoded JPEG XL bytes. |

```python
original_jpeg = pylibjxl.jxl_to_jpeg(jxl_data)
```

---

#### `convert_jpeg_to_jxl(in_path, out_path)` / `convert_jxl_to_jpeg(...)`
File-to-file versions of the above transcoding operations.

```python
pylibjxl.convert_jpeg_to_jxl("input.jpg", "output.jxl")
pylibjxl.convert_jxl_to_jpeg("output.jxl", "restored.jpg")
```

---

### 🏗️ Context Managers

#### `JXL(effort=7, distance=1.0, lossless=False, decoding_speed=0, threads=0, pool_size=0, timeout=None, idle_timeout=30.0)`
#### `AsyncJXL(...)`
Synchronous and Asynchronous context managers maintaining a private, elastic `RunnerPool` with double-layer backpressure and idle reaping.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `effort` | `int` | `7` | Default encoding effort `[1-11]`. |
| `distance` | `float` | `1.0` | Default perceptual distance `[0.0-25.0]`. |
| `lossless` | `bool` | `False` | Default lossless mode. |
| `decoding_speed` | `int` | `0` | Default decoding speed tier `[0-4]`. |
| `threads` | `int` | `0` | Threads per runner (0 = auto-detect). |
| `pool_size` | `int` | `0` | Maximum concurrent runners in the pool (0 = auto-balanced so $M \times N \le \text{Cores}$). |
| `timeout` | `float | None` | `None` | Default acquisition timeout in seconds (`None` = wait indefinitely). |
| `idle_timeout` | `float` | `30.0` | Inactivity duration in seconds before idle runners beyond baseline are reaped. |

#### Read-Only Telemetry Properties
Both `JXL` and `AsyncJXL` expose real-time metrics for health checks and observability:
- `codec.pool_size` (`int`): Maximum allowed runner instances.
- `codec.total_runners` (`int`): Total allocated runner instances currently in memory.
- `codec.available_runners` (`int`): Idle runners ready for immediate acquisition.
- `codec.in_use_runners` (`int`): Runners currently executing encoding/decoding tasks.
- `codec.threads_per_runner` (`int`): Number of internal worker threads per runner.

```python
with pylibjxl.JXL(effort=7, pool_size=4, timeout=5.0) as jxl:
    img = jxl.read("input.jxl")
    jxl.write("output.jxl", img, distance=0.5)
    print(f"Active runners: {jxl.in_use_runners}/{jxl.total_runners}")
```

---

### ℹ️ System Information

| Function | Return Type | Description |
|:---|:---|:---|
| `version()` | `dict` | Returns library version (major, minor, patch). |
| `decoder_version()` | `int` | Returns libjxl decoder version integer. |
| `encoder_version()` | `int` | Returns libjxl encoder version integer. |

```python
print(f"pylibjxl version: {pylibjxl.version()}")
```

---

## 📜 License

[BSD 3-Clause License](LICENSE)
