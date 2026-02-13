# pylibjxl

[![CI](https://github.com/user/pylibjxl/actions/workflows/build.yml/badge.svg)](https://github.com/user/pylibjxl/actions/workflows/build.yml)
[![PyPI version](https://img.shields.io/pypi/v/pylibjxl.svg)](https://pypi.org/project/pylibjxl/)
[![Python versions](https://img.shields.io/pypi/pyversions/pylibjxl.svg)](https://pypi.org/project/pylibjxl/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Fast Python bindings for [libjxl](https://github.com/libjxl/libjxl) and [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo). Built with [pybind11](https://github.com/pybind/pybind11), with GIL-free encoding/decoding and native async support.

## Features

- 🚀 **High performance** — C++ core with GIL release during encode/decode
- 📦 **Metadata support** — Read/write EXIF, XMP, and JUMBF metadata
- ⚡ **Async-first** — Native `asyncio` support for concurrent I/O
- 🎯 **Simple API** — Free functions for quick use, context managers for control
- 🖼️ **NumPy native** — Direct `ndarray` input/output (RGB/RGBA, uint8)
- 🔄 **JPEG support** — Encode/decode JPEG via libjpeg-turbo + lossless JPEG↔JXL transcoding

## Installation

### Prerequisites

- Python ≥ 3.11
- CMake ≥ 3.15
- C++17 compiler (GCC, Clang, MSVC)

> **Note:** libjxl and libjpeg-turbo are bundled as Git submodules in `third_party/` and statically linked — no system-level installation required.

### Install

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv pip install git+https://github.com/user/pylibjxl.git --recursive
```

Or via pip:

```bash
git clone --recurse-submodules https://github.com/user/pylibjxl.git
cd pylibjxl
pip install .
```

## Quick Start

```python
import numpy as np
import pylibjxl

# Create a test image (H, W, C)
image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

# Encode → Decode
data = pylibjxl.encode(image, effort=7, distance=1.0)
decoded = pylibjxl.decode(data)
```

## Usage

### Encode / Decode (In-Memory)

```python
import pylibjxl

# Lossy encoding (default)
data = pylibjxl.encode(image, effort=7, distance=1.0)

# Lossless encoding
data = pylibjxl.encode(image, lossless=True)

# Decode
image = pylibjxl.decode(data)
```

### File I/O

```python
# Write to file (creates parent directories automatically)
pylibjxl.write("output.jxl", image, effort=7, distance=1.0)

# Read from file
image = pylibjxl.read("output.jxl")
```

### Metadata (EXIF / XMP / JUMBF)

```python
# Encode with metadata
data = pylibjxl.encode(image, exif=exif_bytes, xmp=xmp_bytes)

# Decode with metadata extraction
image, meta = pylibjxl.decode(data, metadata=True)
print(meta.keys())  # dict_keys(['exif', 'xmp'])

# File I/O with metadata
pylibjxl.write("photo.jxl", image, exif=exif_bytes, xmp=xmp_bytes, jumbf=jumbf_bytes)
image, meta = pylibjxl.read("photo.jxl", metadata=True)
```

### Context Manager

```python
with pylibjxl.JXL(effort=7, distance=1.0) as jxl:
    # Encode/decode with shared defaults
    data = jxl.encode(image)
    result = jxl.decode(data)

    # Per-call overrides
    data_hq = jxl.encode(image, distance=0.5)

    # File I/O
    jxl.write("output.jxl", image, exif=exif_bytes)
    result, meta = jxl.read("output.jxl", metadata=True)
```

### Async

```python
import asyncio
import pylibjxl

async def main():
    # In-memory async
    data = await pylibjxl.encode_async(image, exif=exif_bytes)
    image, meta = await pylibjxl.decode_async(data, metadata=True)

    # File async
    await pylibjxl.write_async("output.jxl", image, xmp=xmp_bytes)
    image = await pylibjxl.read_async("output.jxl")

    # Async context manager
    async with pylibjxl.AsyncJXL(effort=5) as jxl:
        data = await jxl.encode_async(image)
        result = await jxl.decode_async(data)

asyncio.run(main())
```

### JPEG Encode / Decode

```python
import pylibjxl

# Encode to JPEG (via libjpeg-turbo)
jpeg_data = pylibjxl.encode_jpeg(image, quality=95)

# Decode JPEG to numpy array
image = pylibjxl.decode_jpeg(jpeg_data)
```

### JPEG ↔ JXL Transcoding

```python
# Losslessly recompress JPEG → JXL (preserves JPEG reconstruction data)
jxl_data = pylibjxl.jpeg_to_jxl(jpeg_data, effort=7)

# Reconstruct original JPEG from JXL (lossless roundtrip)
jpeg_restored = pylibjxl.jxl_to_jpeg(jxl_data)

# Async variants
jxl_data = await pylibjxl.jpeg_to_jxl_async(jpeg_data)
jpeg_data = await pylibjxl.jxl_to_jpeg_async(jxl_data)
```

### JPEG File I/O

```python
# Write JPEG file
pylibjxl.write_jpeg("photo.jpg", image, quality=95)

# Read JPEG file
image = pylibjxl.read_jpeg("photo.jpg")

# Async
await pylibjxl.write_jpeg_async("photo.jpg", image)
image = await pylibjxl.read_jpeg_async("photo.jpg")
```

### Cross-Format File Conversion

```python
# JPEG → JXL (lossless transcoding, preserves JPEG reconstruction data)
pylibjxl.convert_jpeg_to_jxl("photo.jpg", "photo.jxl")

# JXL → JPEG (lossless reconstruction from transcoded JXL)
pylibjxl.convert_jxl_to_jpeg("photo.jxl", "restored.jpg")

# Async
await pylibjxl.convert_jpeg_to_jxl_async("photo.jpg", "photo.jxl")
await pylibjxl.convert_jxl_to_jpeg_async("photo.jxl", "restored.jpg")
```

## API Reference

### Free Functions

#### `encode(input, effort=7, distance=1.0, lossless=False, *, exif=None, xmp=None, jumbf=None) → bytes`

Encode a NumPy array to JXL bytes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `ndarray` | *required* | uint8 array of shape `(H, W, 3)` or `(H, W, 4)` |
| `effort` | `int` | `7` | Encoding effort `[1-10]`, higher = slower + smaller |
| `distance` | `float` | `1.0` | Perceptual distance `[0.0-25.0]`, `0` = lossless |
| `lossless` | `bool` | `False` | If `True`, encode losslessly (overrides distance) |
| `exif` | `bytes \| None` | `None` | Raw EXIF metadata to embed |
| `xmp` | `bytes \| None` | `None` | Raw XMP (XML) metadata to embed |
| `jumbf` | `bytes \| None` | `None` | Raw JUMBF metadata to embed |

> **Note on EXIF:** `pylibjxl` automatically handles the 4-byte TIFF header offset required by the JXL box format. You should provide raw EXIF bytes starting with the TIFF header (e.g., `II*` or `MM*`).

---

#### `decode(data, *, metadata=False) → ndarray | tuple[ndarray, dict]`

Decode JXL bytes to a NumPy array.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `bytes` | *required* | JXL-encoded data |
| `metadata` | `bool` | `False` | If `True`, also return metadata dict |

**Returns:**
- `metadata=False` → `ndarray` of shape `(H, W, C)`, dtype `uint8`
- `metadata=True` → `tuple(ndarray, dict)` where dict may contain keys: `"exif"`, `"xmp"`, `"jumbf"` (as `bytes`)

---

#### `read(path, *, metadata=False)`

Read a `.jxl` file from disk. Returns same types as `decode()`.

#### `write(path, image, effort=7, distance=1.0, lossless=False, *, exif=None, xmp=None, jumbf=None)`

Encode and write to a `.jxl` file. Creates parent directories automatically.

---

#### `encode_async(...)` / `decode_async(...)` / `read_async(...)` / `write_async(...)`

Async versions of the above functions — same parameters, returns `Awaitable`.

---

### Context Managers

#### `JXL(effort=7, distance=1.0, lossless=False)`

Synchronous codec context manager with shared defaults. It maintains a persistent thread pool for better performance across multiple operations.

| Method | Description |
|--------|-------------|
| `encode(input, ...)` | Encode JXL in-memory (supports per-call overrides) |
| `decode(data, *, metadata=False)` | Decode JXL in-memory |
| `read(path, *, metadata=False)` | Read JXL from file |
| `write(path, image, ...)` | Write JXL to file |
| `encode_jpeg(input, quality=95)` | Encode JPEG in-memory |
| `decode_jpeg(data)` | Decode JPEG in-memory |
| `read_jpeg(path)` | Read JPEG from file |
| `write_jpeg(path, image, quality=95)` | Write JPEG to file |
| `jpeg_to_jxl(data, effort=None)` | Lossless JPEG → JXL transcoding |
| `jxl_to_jpeg(data)` | JXL → JPEG reconstruction |
| `convert_jpeg_to_jxl(in_path, out_path)` | File-to-file JPEG → JXL |
| `convert_jxl_to_jpeg(in_path, out_path)` | File-to-file JXL → JPEG |
| `close()` | Explicitly close and release thread pool |

**Properties:**
- `closed` (bool): Whether the codec context has been closed.

#### `AsyncJXL(effort=7, distance=1.0, lossless=False)`

Async codec context manager. Methods are async versions of the above:
`encode_async`, `decode_async`, `read_async`, `write_async`, `encode_jpeg_async`, `decode_jpeg_async`, `read_jpeg_async`, `write_jpeg_async`, `jpeg_to_jxl_async`, `jxl_to_jpeg_async`, `convert_jpeg_to_jxl_async`, `convert_jxl_to_jpeg_async`.

---

### JPEG & Transcoding Functions

#### `encode_jpeg(input, quality=95) → bytes`

Encode a NumPy array to JPEG bytes using libjpeg-turbo.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `ndarray` | *required* | uint8 array of shape `(H, W, 3)` or `(H, W, 4)` |
| `quality` | `int` | `95` | JPEG quality `[1-100]` |

#### `decode_jpeg(data) → ndarray`

Decode JPEG bytes to a NumPy array `(H, W, 3)` using libjpeg-turbo.

#### `read_jpeg(path) → ndarray`

Read a `.jpg`/`.jpeg` file from disk. Returns `ndarray` of shape `(H, W, 3)`.

#### `write_jpeg(path, image, quality=95)`

Encode and write to a JPEG file. Creates parent directories automatically.

#### `jpeg_to_jxl(data, effort=7) → bytes`

Losslessly recompress JPEG bytes to JXL. This process preserves the original JPEG codestream and metadata (EXIF, XMP, etc.), allowing for bit-perfect restoration of the original JPEG file.

#### `jxl_to_jpeg(data) → bytes`

Reconstruct the original JPEG bytes from a JXL file (only works if the JXL was created via `jpeg_to_jxl`).

#### `convert_jpeg_to_jxl(jpeg_path, jxl_path, effort=7)`

Convert a JPEG file to JXL file via lossless transcoding. Creates parent directories automatically.

#### `convert_jxl_to_jpeg(jxl_path, jpeg_path)`

Reconstruct the original JPEG file from a JXL file. Only works for JPEG-transcoded JXL files.

#### Async variants

`encode_jpeg_async`, `decode_jpeg_async`, `read_jpeg_async`, `write_jpeg_async`, `jpeg_to_jxl_async`, `jxl_to_jpeg_async`, `convert_jpeg_to_jxl_async`, `convert_jxl_to_jpeg_async` — same parameters, returns `Awaitable`.

---

### Utility Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `version()` | `dict` | libjxl version `{"major", "minor", "patch"}` |
| `decoder_version()` | `int` | Decoder version number |
| `encoder_version()` | `int` | Encoder version number |

## License

BSD 3-Clause
