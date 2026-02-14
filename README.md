<div align="center">

# pylibjxl

**Fast Python bindings for JPEG XL (libjxl) and JPEG (libjpeg-turbo)**

[![CI](https://github.com/twn39/pylibjxl/actions/workflows/build.yml/badge.svg)](https://github.com/twn39/pylibjxl/actions/workflows/build.yml)
[![PyPI version](https://img.shields.io/pypi/v/pylibjxl.svg)](https://pypi.org/project/pylibjxl/)
[![Python versions](https://img.shields.io/pypi/pyversions/pylibjxl.svg)](https://pypi.org/project/pylibjxl/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

---

`pylibjxl` provides efficient, high-performance Python bindings for [libjxl](https://github.com/libjxl/libjxl) and [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo). Built with [pybind11](https://github.com/pybind/pybind11), it features **GIL-free** encoding/decoding and **native async** support for maximum throughput.

</div>

## ✨ Key Features

- 🚀 **High Performance** — C++ core releases the GIL during heavy computation.
- 📦 **Metadata Excellence** — Full support for EXIF, XMP, and JUMBF metadata.
- ⚡ **Async-First** — Native `asyncio` integration for non-blocking I/O.
- 🖼️ **NumPy Native** — Directly encode from and decode to `ndarray` (RGB/RGBA).
- 🔄 **Lossless JPEG Transcoding** — Bit-perfect JPEG ↔ JXL roundtrips.
- 🎯 **Thread-Safe** — Persistent thread pools via context managers.

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

---

## 📈 Performance & Stability

`pylibjxl` is designed for high-throughput production environments:

- **GIL-Free**: C++ core releases the Global Interpreter Lock during encoding/decoding, enabling true multi-core parallelism via Python's `threading` or `asyncio`.
- **Memory Efficient**: Uses `libjxl`'s internal memory management and pre-allocates buffers to minimize reallocations. 
- **Stable Throughput**: Benchmarks show consistent timing with very low jitter (Standard Deviation < 1ms for decoding) over thousands of operations.
- **Thread Pool Reuse**: Using the `JXL` context manager prevents the overhead of creating/destroying thread pools for every operation, providing an additional ~15% speedup in batch processing.

### Benchmark (1440x960 RGB)
| Library | Task | Mean Time |
|:---|:---|:---|
| **pylibjxl** | **JXL Encode (effort 7)** | **~105 ms** |
| pillow-jxl-plugin | JXL Encode (effort 7) | ~95 ms |
| **pylibjxl** | **JXL Decode** | **10.8 ms** |
| pillow-jxl-plugin | JXL Decode | 11.6 ms |

> [!TIP]
> While single-shot synchronous performance is comparable to other implementations, **pylibjxl**'s core strength lies in its **GIL-free** architecture and **native async** support. This ensures your application remains responsive and scales linearly across multiple CPU cores during high-concurrency workloads—something standard Pillow-based plugins cannot achieve due to the Global Interpreter Lock.

---

## 📂 API Reference

### 🖼️ JXL In-Memory Operations

#### `encode(input, effort=7, distance=1.0, lossless=False, *, exif=None, xmp=None, jumbf=None) -> bytes`
#### `async encode_async(...) -> bytes`
Encodes a NumPy array into JPEG XL format.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `ndarray` | *required* | uint8 array of shape `(H, W, 3)` or `(H, W, 4)` |
| `effort` | `int` | `7` | Speed/size tradeoff `[1-10]`. 1=fastest, 10=best compression. |
| `distance` | `float` | `1.0` | Perceptual quality `[0.0-25.0]`. 0.0=lossless, 1.0=visually lossless. |
| `lossless` | `bool` | `False` | If `True`, enables mathematical lossless mode. |
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
| `effort` | `int` | `7` | Transcoding effort `[1-10]`. |

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

#### `JXL(effort=7, distance=1.0, lossless=False)`
#### `AsyncJXL(...)`
Sync and Async context managers that maintain a persistent thread pool.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `effort` | `int` | `7` | Default effort for operations. |
| `distance` | `float` | `1.0` | Default distance for operations. |
| `lossless` | `bool` | `False` | Default lossless mode. |

```python
with pylibjxl.JXL(effort=7) as jxl:
    # Uses persistent threads for all methods
    img = jxl.read("input.jxl")
    jxl.write("output.jxl", img, distance=0.5)
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
