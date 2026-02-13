# pylibjxl

Fast Python bindings for [libjxl](https://github.com/libjxl/libjxl) — the JPEG XL reference implementation. Built with [pybind11](https://github.com/pybind/pybind11), with GIL-free encoding/decoding and native async support.

## Features

- 🚀 **High performance** — C++ core with GIL release during encode/decode
- 📦 **Metadata support** — Read/write EXIF, XMP, and JUMBF metadata
- ⚡ **Async-first** — Native `asyncio` support for concurrent I/O
- 🎯 **Simple API** — Free functions for quick use, context managers for control
- 🖼️ **NumPy native** — Direct `ndarray` input/output (RGB/RGBA, uint8)

## Installation

### Prerequisites

- Python ≥ 3.11
- CMake ≥ 3.15
- C++17 compiler (GCC, Clang, MSVC)

> **Note:** libjxl is bundled as a Git submodule in `third_party/libjxl` and statically linked — no system-level installation required.

### Install

```bash
git clone --recurse-submodules https://github.com/user/pylibjxl.git
cd pylibjxl
```

```bash
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

Synchronous codec context manager with shared defaults.

| Method | Description |
|--------|-------------|
| `encode(input, effort=None, distance=None, lossless=None, *, exif=None, xmp=None, jumbf=None)` | Encode in-memory |
| `decode(data, *, metadata=False)` | Decode in-memory |
| `read(path, *, metadata=False)` | Read from file |
| `write(path, image, effort=None, distance=None, lossless=None, *, exif=None, xmp=None, jumbf=None)` | Write to file |

#### `AsyncJXL(effort=7, distance=1.0, lossless=False)`

Async codec context manager. Methods: `encode_async`, `decode_async`, `read_async`, `write_async`.

---

### Utility Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `version()` | `dict` | libjxl version `{"major", "minor", "patch"}` |
| `decoder_version()` | `int` | Decoder version number |
| `encoder_version()` | `int` | Encoder version number |

## License

MIT
