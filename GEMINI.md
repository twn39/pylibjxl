# GEMINI.md - pylibjxl

## Project Overview

`pylibjxl` provides high-performance Python bindings for **JPEG XL (libjxl)** and **JPEG (libjpeg-turbo)**. It is designed for speed, efficiency, and seamless integration with the Python ecosystem, particularly for heavy image processing tasks.

### Key Technologies
- **C++ Core**: Uses `pybind11` for bindings and releases the Python Global Interpreter Lock (GIL) during heavy computation to enable true multi-core parallelism.
- **Python Layer**: Provides high-level APIs, including native `asyncio` support via `asyncio.to_thread`.
- **Image Data**: Uses `numpy.ndarray` (uint8) as the primary image representation.
- **Build System**: Powered by `scikit-build-core` and `CMake`.
- **Submodules**: Bundles `libjxl` and `libjpeg-turbo` as git submodules.

## Architecture

- **`src/main.cpp`**: The C++ entry point. Defines the `_pylibjxl` extension module.
    - Implements GIL-free encoding and decoding.
    - Uses `JxlResizableParallelRunner` with `thread_local` storage for efficient multi-threading.
    - Handles EXIF, XMP, and JUMBF metadata boxes.
- **`src/pylibjxl/__init__.py`**: The Python wrapper.
    - Maps low-level C++ functions to a user-friendly API.
    - Implements `encode_async`, `decode_async`, and other `_async` variants using `asyncio.to_thread`.
    - Provides `JXL` (sync) and `AsyncJXL` (async) context managers for persistent thread pool reuse, allowing explicit control over worker threads to prevent resource exhaustion in concurrent environments.
- **`third_party/`**: Contains submodules for `libjxl` and `libjpeg-turbo`.

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

## Development Conventions

### GIL Management
Always release the GIL in C++ for any operation that takes significant time (encoding, decoding, transcoding). This allows Python's threading to work effectively.

### Async Patterns
Prefer `asyncio.to_thread` in the Python layer for I/O and CPU-bound tasks that release the GIL, ensuring the event loop remains responsive.

### Metadata Handling
Support for EXIF, XMP, and JUMBF should be maintained. JXL metadata is handled via boxes. Note that `libjxl` requires a 4-byte prefix for EXIF boxes which the C++ core handles automatically.

### Code Quality
- **Linting**: Uses `ruff` (configured in `pyproject.toml`).
- **Typing**: Uses `pyright` for type checking.
- **Formatting**: C++ code follows `.clang-format`.
- **CI**: Build and tests are automated via GitHub Actions (`.github/workflows/build.yml`).
