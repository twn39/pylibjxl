# GEMINI.md - pylibjxl

## Project Overview

`pylibjxl` provides high-performance Python bindings for **JPEG XL (libjxl)** and **JPEG (libjpeg-turbo)**. It is designed for speed, efficiency, and seamless integration with the Python ecosystem, particularly for heavy image processing tasks.

### Key Technologies
- **C++ Core**: Uses `nanobind` for bindings and releases the Python Global Interpreter Lock (GIL) during heavy computation to enable true multi-core parallelism.
- **Python Layer**: Provides high-level APIs, including native `asyncio` support via `asyncio.to_thread`.
- **Image Data**: Uses `numpy.ndarray` (uint8) as the primary image representation.
- **Build System**: Powered by `scikit-build-core` and `CMake`.
- **Submodules**: Bundles `libjxl` and `libjpeg-turbo` as git submodules.

## Architecture

- **`src/main.cpp`**: The C++ entry point. Defines the `_pylibjxl` extension module.
    - Implements GIL-free encoding and decoding.
    - Uses `RunnerPool` — a thread-safe pool of `JxlResizableParallelRunner` instances for true concurrent parallelism.
    - `RunnerGuard` provides RAII-based acquire/release of runners from the pool.
    - Free functions (`encode`, `decode`, etc.) use a lazily-initialized global `RunnerPool`.
    - Handles EXIF, XMP, and JUMBF metadata boxes.
- **`src/pylibjxl/__init__.py`**: The Python wrapper.
    - Maps low-level C++ functions to a user-friendly API.
    - Implements `encode_async`, `decode_async`, and other `_async` variants using `asyncio.to_thread`.
    - Provides `JXL` (sync) and `AsyncJXL` (async) context managers, each owning a private `RunnerPool` for controlled resource usage.
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

### Concurrency Model (RunnerPool)
`JxlResizableParallelRunner` is **not thread-safe** — two concurrent operations cannot share the same runner. `RunnerPool` solves this by maintaining a pool of independent runners:
- **`RunnerPool(pool_size, threads_per_runner)`**: Creates `pool_size` runner instances, each with `threads_per_runner` internal threads.
- **`acquire()` / `release()`**: Thread-safe borrow/return operations. `acquire()` blocks via `condition_variable` if no runners are available.
- **`RunnerGuard`**: RAII wrapper that calls `acquire()` on construction and `release()` on destruction, ensuring exception safety.
- **Global pool** (`global_pool()`): Lazily initialized with `pool_size = hardware_concurrency()` and `threads_per_runner = 1`, used by free functions.
- **`PyJxlCodec` pool**: Each `JXL`/`AsyncJXL` instance owns a private pool, allowing true parallel encode/decode within a single context manager.

### Async Patterns
Prefer `asyncio.to_thread` in the Python layer for I/O and CPU-bound tasks that release the GIL, ensuring the event loop remains responsive.

### Metadata Handling
Support for EXIF, XMP, and JUMBF should be maintained. JXL metadata is handled via boxes. Note that `libjxl` requires a 4-byte prefix for EXIF boxes which the C++ core handles automatically.

### Code Quality
- **Linting**: Uses `ruff` (configured in `pyproject.toml`).
- **Typing**: Uses `pyright` for type checking.
- **Formatting**: C++ code follows `.clang-format`.
- **CI**: Build and tests are automated via GitHub Actions (`.github/workflows/build.yml`).
