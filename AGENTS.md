# AGENTS.md - pylibjxl

## Guidelines for AI Agents (Antigravity, Claude Code, Cursor, Roo Code, etc.)

### Codebase Knowledge Graph (`.codegraph/`)

This project maintains a codebase knowledge graph at `.codegraph/`. You MUST follow these rules when working in this codebase:

1. **Prioritize the Knowledge Graph**:
   - Before answering architecture, design, or codebase structure questions, you **MUST** read [.codegraph/README.md](.codegraph/README.md) to understand the system overview, god nodes, and logical community structure.
   - Use [.codegraph/components/](.codegraph/components/) and [.codegraph/nodes/](.codegraph/nodes/) to navigate component boundaries, file relationships, and symbol definitions. This is much faster and more token-efficient than reading raw source files directly.

2. **AI Architectural Insights**:
   - Check [.codegraph/README.md](.codegraph/README.md) for a section titled `AI Architectural Insights`.
   - If this section is missing, incomplete, or contains placeholders, read [.codegraph/AGENT_PROMPT.md](.codegraph/AGENT_PROMPT.md), perform a deep architectural analysis of the project, and write your report into that section. Do not overwrite other sections.

3. **Keep Graph Sync'd**:
   - Whenever you create, delete, or modify code files, you **SHOULD** remind the user to run `codegraph build .` to rebuild the knowledge graph and keep it current.
   - When running the build command, exclude irrelevant or generated directories (e.g., third-party dependencies, build folders, or documentation) using the `-e`/`--exclude` flag to keep the graph focused and clean (e.g., `codegraph build . -e third_party/ -e build/ -e dist/`).

---

## Project Overview

`pylibjxl` provides high-performance Python bindings for **JPEG XL (libjxl)** and **JPEG (libjpeg-turbo)**. It is designed for speed, efficiency, and seamless integration with the Python ecosystem, particularly for heavy image processing tasks.

### Key Technologies
- **C++ Core**: Uses `nanobind` for bindings and releases the Python Global Interpreter Lock (GIL) during heavy computation to enable true multi-core parallelism.
- **Python Layer**: Provides high-level APIs, including native `asyncio` support via `asyncio.to_thread`.
- **Color Management**: Links `jxl_cms` with `skcms` integration for full ICC color profile support.
- **Image Data**: Uses `numpy.ndarray` (uint8) as the primary image representation.
- **Build System**: Powered by `scikit-build-core` and `CMake`.
- **Submodules**: Bundles `libjxl`, `libjpeg-turbo`, and `nanobind` as git submodules in `third_party/`.

---

## Architecture

### C++ Core (`src/`)
The native extension `_pylibjxl` is organized into clean, modular sub-components:
- **`src/common/`**:
  - `deleters.hpp`: RAII smart deleters and pointer aliases for `libjxl` and `libjpeg-turbo` handles (`JxlEncoderPtr`, `JxlDecoderPtr`, `JxlRunnerPtr`, `TjPtr`, `TjBufPtr`).
  - `utils.hpp`: Common C++ helpers (`extract_optional_bytes`).
- **`src/concurrency/`**:
  - `runner_pool.hpp` / `runner_pool.cpp`: Thread-safe `RunnerPool` maintaining pools of independent `JxlResizableParallelRunner` instances, RAII `RunnerGuard`, and lazily-initialized `global_pool()`.
- **`src/codecs/`**:
  - `jxl_ops.hpp` / `jxl_ops.cpp`: GIL-free JXL encoding and decoding, EXIF/XMP/JUMBF box handling, and ICC profile extraction/injection with `JxlGetDefaultCms()`.
  - `jpeg_ops.hpp` / `jpeg_ops.cpp`: GIL-free TurboJPEG encoding and decoding for RGB/RGBA buffers.
  - `transcode.hpp` / `transcode.cpp`: Fast, lossless cross-format transcoding (`jpeg_to_jxl`, `jxl_to_jpeg`).
- **`src/bindings/`**:
  - `py_codec.hpp`: `PyJxlCodec` class definition managing private thread pools for context managers.
  - `module.cpp`: Nanobind module definition exposing public C++ functions and types.

### Python Layer (`src/pylibjxl/`)
- **`src/pylibjxl/__init__.py`**: Clean unified entrypoint exposing the public API (`__all__`).
- **`src/pylibjxl/_io.py`**: Synchronous file I/O operations (`read`, `write`, `read_jpeg`, `write_jpeg`, `convert_jpeg_to_jxl`, `convert_jxl_to_jpeg`).
- **`src/pylibjxl/_async.py`**: Asynchronous variants using `asyncio.to_thread` for non-blocking I/O and parallel execution.
- **`src/pylibjxl/_context.py`**: Context managers `JXL` (synchronous) and `AsyncJXL` (asynchronous) providing isolated runner pools and convenient batch/stream APIs.
- **`src/pylibjxl/__init__.pyi`**: Complete PEP 484 type annotations and overloads for IDE autocompletion and static analysis.
- **`third_party/`**: Contains submodules for `libjxl`, `libjpeg-turbo`, and `nanobind`.

---

## Building and Running

### Development Environment Setup
This project uses `uv` for dependency management.
```bash
# Clone with submodules
git clone --recursive https://github.com/twn39/pylibjxl.git
cd pylibjxl

# Install development dependencies
uv pip install -e ".[dev]"
```

### Build Commands
```bash
# Standard editable install
uv pip install -e .

# Build wheels or sdist
python -m build
```

### Testing
Tests are located in the `tests/` directory and use `pytest`.
```bash
# Run all tests
uv run pytest

# Run benchmarks
uv run pytest --benchmark-only

# Run with coverage
uv run pytest --cov=pylibjxl
```

---

## Development Conventions

### GIL Management
Always release the GIL in C++ for any operation that takes significant time (encoding, decoding, transcoding). This allows Python's threading and `asyncio.to_thread` to work effectively.

### Concurrency Model (RunnerPool)
`JxlResizableParallelRunner` is **not thread-safe** — two concurrent operations cannot share the same runner. `RunnerPool` solves this by maintaining a pool of independent runners:
- **`RunnerPool(pool_size, threads_per_runner)`**: Creates `pool_size` runner instances, each with `threads_per_runner` internal threads.
- **`acquire()` / `release()`**: Thread-safe borrow/return operations. `acquire()` blocks via `condition_variable` if no runners are available.
- **`RunnerGuard`**: RAII wrapper that calls `acquire()` on construction and `release()` on destruction, ensuring exception safety.
- **Global pool** (`global_pool()`): Lazily initialized with `pool_size = hardware_concurrency()` and `threads_per_runner = 1`, used by free functions.
- **`PyJxlCodec` pool**: Each `JXL`/`AsyncJXL` instance owns a private pool, allowing true parallel encode/decode within a single context manager.

### Async Patterns
Prefer `asyncio.to_thread` in the Python layer for I/O and CPU-bound tasks that release the GIL, ensuring the event loop remains responsive.

### Metadata & Color Profile Handling
- **EXIF, XMP, JUMBF**: Stored as JXL container boxes. Note that `libjxl` requires a 4-byte prefix for EXIF boxes which the C++ core handles automatically.
- **ICC Profiles**:
  - Encoded via `JxlEncoderSetICCProfile` when `icc` bytes are provided; defaults to sRGB (`JxlColorEncodingSetToSRGB`) with channel-aware grayscale handling.
  - Decoded via `JxlDecoderGetColorAsICCProfile(..., JXL_COLOR_PROFILE_TARGET_ORIGINAL, ...)` with default CMS registered (`JxlDecoderSetCms(dec.get(), *JxlGetDefaultCms())`).
  - Decoded metadata dictionary returns both `"icc"` and `"icc_profile"` for Pillow ecosystem compatibility when an ICC profile is present.

### Zero-Copy Buffer Protocol & In-Place Decode
- **Buffer Protocol (Ingress)**: `decode`, `decode_jpeg`, `jpeg_to_jxl`, and `jxl_to_jpeg` accept any Python object supporting the buffer protocol (`bytes`, `bytearray`, `memoryview`, `mmap.mmap`, `ndarray`) using `ScopedPyBuffer` without copying data.
- **In-Place Output (Egress)**: `decode` and `decode_jpeg` support an optional `out` parameter (C-contiguous uint8 `numpy.ndarray`), allowing zero-allocation in-place decoding.
- **Memory-Mapped I/O**: `read` and `read_jpeg` support `use_mmap=True` for direct OS page-cache streaming into the decoder.
- **Direct File Transcoding**: `jpeg_to_jxl_file` and `jxl_to_jpeg_file` operate entirely in C++ without allocating intermediate Python heap objects.

### Code Quality
- **Linting**: Uses `ruff` (configured in `pyproject.toml`).
- **Typing**: Uses `pyright` for type checking.
- **Formatting**: C++ code follows `.clang-format`.
- **CI**: Build and tests are automated via GitHub Actions (`.github/workflows/build.yml`).

## codegraph-gen

This project maintains a codebase knowledge graph at `.codegraph/`.

### Guidelines for AI Agents (Antigravity, Claude Code, Cursor, Roo Code, etc.)

You MUST follow these rules when working in this codebase:

1. **Prioritize the Knowledge Graph**:
   - Before answering architecture, design, or codebase structure questions, you **MUST** read [.codegraph/README.md](.codegraph/README.md) to understand the system overview, god nodes, and logical community structure.
   - Use [.codegraph/components/](.codegraph/components/) and [.codegraph/nodes/](.codegraph/nodes/) to navigate component boundaries, file relationships, and symbol definitions. This is much faster and more token-efficient than reading raw source files directly.

2. **AI Architectural Insights**:
   - Check [.codegraph/README.md](.codegraph/README.md) for a section titled `AI Architectural Insights`.
   - If this section is missing, incomplete, or contains placeholders, read [.codegraph/AGENT_PROMPT.md](.codegraph/AGENT_PROMPT.md), perform a deep architectural analysis of the project, and write your report into that section. Do not overwrite other sections.

3. **Keep Graph Sync'd**:
   - Whenever you create, delete, or modify code files, you **SHOULD** remind the user to run `codegraph build .` to rebuild the knowledge graph and keep it current.
   - When running the build command, exclude irrelevant or generated directories (e.g., third-party dependencies, build folders, or documentation) using the `-e`/`--exclude` flag to keep the graph focused and clean (e.g., `codegraph build . -e third_party/`).
