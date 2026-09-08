import asyncio
import mmap
from pathlib import Path

import numpy as np

from ._pylibjxl import JXL as _JXL  # type: ignore
from ._pylibjxl import CodecTimeoutError  # type: ignore


class JXL(_JXL):
    """Unified JXL/JPEG codec with synchronous context manager support.

    Owns an elastic thread pool that is destroyed on close()/exit.
    Supports JXL encode/decode, JPEG encode/decode, cross-format
    transcoding, and file I/O for both formats.

    Usage::

        with pylibjxl.JXL(effort=7, pool_size=4, timeout=5.0) as jxl:
            # JXL
            jxl.write("output.jxl", image, exif=exif_bytes)
            img, meta = jxl.read("output.jxl", metadata=True)
            # In-place decode
            out_buf = np.empty_like(img)
            jxl.decode(data, out=out_buf)
            # JPEG
            jxl.write_jpeg("output.jpg", image, quality=95)
            img2 = jxl.read_jpeg("output.jpg")
            # Cross-format direct file transcoding
            jxl.convert_jpeg_to_jxl("input.jpg", "output.jxl")
            jxl.convert_jxl_to_jpeg("input.jxl", "output.jpg")

    Args:
        effort: Encoding effort [1-11] (default 7).
        distance: Perceptual distance [0.0-25.0] (default 1.0).
        lossless: If True, encode losslessly (default False).
        decoding_speed: Decoding speed tier [0-4] (default 0).
        threads: Number of worker threads per runner (default 0 = auto-detect).
        pool_size: Maximum concurrent operations (default 0 = auto-balance).
        timeout: Acquisition timeout in seconds (default None = wait indefinitely).
        idle_timeout: Idle runner reap timeout in seconds (default 30.0).
    """

    def __init__(
        self,
        effort=7,
        distance=1.0,
        lossless=False,
        decoding_speed=0,
        threads=0,
        pool_size=0,
        timeout=None,
        idle_timeout=30.0,
    ):
        super().__init__(
            effort=effort,
            distance=distance,
            lossless=lossless,
            decoding_speed=decoding_speed,
            threads=threads,
            pool_size=pool_size,
            timeout=timeout,
            idle_timeout=idle_timeout,
        )

    def encode(
        self,
        input,
        effort=None,
        distance=None,
        lossless=None,
        decoding_speed=None,
        exif=None,
        xmp=None,
        jumbf=None,
        icc=None,
        *,
        timeout=None,
    ):
        if hasattr(input, "flags") and not input.flags.c_contiguous:
            input = np.ascontiguousarray(input)
        return super().encode(
            input,
            effort,
            distance,
            lossless,
            decoding_speed,
            exif,
            xmp,
            jumbf,
            icc,
            timeout=timeout,
        )

    def decode(self, data, *, metadata=False, out=None, timeout=None):
        return super().decode(data, metadata=metadata, out=out, timeout=timeout)

    def encode_jpeg(self, input, quality=95):
        if hasattr(input, "flags") and not input.flags.c_contiguous:
            input = np.ascontiguousarray(input)
        return super().encode_jpeg(input, quality=quality)

    # ── JXL File I/O ──

    def read(self, path, *, metadata=False, out=None, use_mmap=False, timeout=None):
        """Read a JXL file and return a numpy array."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"No such file: '{filepath}'")
        if use_mmap:
            with open(filepath, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    return self.decode(mm, metadata=metadata, out=out, timeout=timeout)
        data = filepath.read_bytes()
        return self.decode(data, metadata=metadata, out=out, timeout=timeout)

    def write(
        self,
        path,
        image,
        effort=None,
        distance=None,
        lossless=None,
        decoding_speed=None,
        *,
        exif=None,
        xmp=None,
        jumbf=None,
        icc=None,
        timeout=None,
    ):
        """Encode a numpy array and write it to a JXL file."""
        if hasattr(image, "flags") and not image.flags.c_contiguous:
            image = np.ascontiguousarray(image)
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = self.encode(
            image,
            effort,
            distance,
            lossless,
            decoding_speed,
            exif,
            xmp,
            jumbf,
            icc,
            timeout=timeout,
        )
        filepath.write_bytes(data)

    # ── JPEG File I/O ──

    def read_jpeg(self, path, *, out=None, use_mmap=False):
        """Read a JPEG file and return a numpy array (H, W, 3)."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"No such file: '{filepath}'")
        if use_mmap:
            with open(filepath, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    return self.decode_jpeg(mm, out=out)
        data = filepath.read_bytes()
        return self.decode_jpeg(data, out=out)

    def write_jpeg(self, path, image, quality=95):
        """Encode a numpy array and write it to a JPEG file."""
        if hasattr(image, "flags") and not image.flags.c_contiguous:
            image = np.ascontiguousarray(image)
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = self.encode_jpeg(image, quality=quality)
        filepath.write_bytes(data)

    # ── Cross-Format File Conversion ──

    def convert_jpeg_to_jxl(self, jpeg_path, jxl_path, effort=None, *, timeout=None):
        """Convert a JPEG file to JXL file (lossless transcoding)."""
        jpeg_filepath = Path(jpeg_path)
        if not jpeg_filepath.exists():
            raise FileNotFoundError(f"No such file: '{jpeg_filepath}'")
        jxl_filepath = Path(jxl_path)
        jxl_filepath.parent.mkdir(parents=True, exist_ok=True)
        self.jpeg_to_jxl_file(
            str(jpeg_filepath), str(jxl_filepath), effort=effort, timeout=timeout
        )

    def convert_jxl_to_jpeg(self, jxl_path, jpeg_path, *, timeout=None):
        """Convert a JXL file to JPEG file (lossless reconstruction)."""
        jxl_filepath = Path(jxl_path)
        if not jxl_filepath.exists():
            raise FileNotFoundError(f"No such file: '{jxl_filepath}'")
        jpeg_filepath = Path(jpeg_path)
        jpeg_filepath.parent.mkdir(parents=True, exist_ok=True)
        self.jxl_to_jpeg_file(str(jxl_filepath), str(jpeg_filepath), timeout=timeout)


class AsyncJXL(JXL):
    """Unified JXL/JPEG codec with async context manager support.

    Owns an elastic thread pool with asyncio.Semaphore backpressure.

    Usage::

        async with pylibjxl.AsyncJXL(effort=7, pool_size=4, timeout=5.0) as jxl:
            await jxl.write_async("output.jxl", image)
            await jxl.write_jpeg_async("output.jpg", image)
            await jxl.convert_jpeg_to_jxl_async("in.jpg", "out.jxl")

    Args:
        effort: Encoding effort [1-11] (default 7).
        distance: Perceptual distance [0.0-25.0] (default 1.0).
        lossless: If True, encode losslessly (default False).
        decoding_speed: Decoding speed tier [0-4] (default 0).
        threads: Number of worker threads per runner (default 0 = auto-detect).
        pool_size: Maximum concurrent operations (default 0 = auto-balance).
        timeout: Acquisition timeout in seconds (default None = wait indefinitely).
        idle_timeout: Idle runner reap timeout in seconds (default 30.0).
    """

    def __init__(
        self,
        effort=7,
        distance=1.0,
        lossless=False,
        decoding_speed=0,
        threads=0,
        pool_size=0,
        timeout=None,
        idle_timeout=30.0,
    ):
        super().__init__(
            effort=effort,
            distance=distance,
            lossless=lossless,
            decoding_speed=decoding_speed,
            threads=threads,
            pool_size=pool_size,
            timeout=timeout,
            idle_timeout=idle_timeout,
        )
        self._default_timeout = timeout
        self._semaphore = (
            asyncio.Semaphore(self.pool_size) if self.pool_size > 0 else None
        )

    async def __aenter__(self):
        self.__enter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__exit__(exc_type, exc_val, exc_tb)

    async def _run_guarded(self, func, *args, **kwargs):
        timeout = kwargs.get("timeout")
        if (
            timeout is None
            and self._default_timeout is not None
            and self._default_timeout > 0
        ):
            timeout = self._default_timeout

        if self._semaphore is not None:
            if timeout is not None and timeout > 0:
                t0 = asyncio.get_running_loop().time()
                try:
                    await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
                except asyncio.TimeoutError:
                    raise CodecTimeoutError(
                        f"RunnerPool acquisition timed out after {int(timeout * 1000)}ms"
                    ) from None
                try:
                    elapsed = asyncio.get_running_loop().time() - t0
                    remaining = max(0.001, timeout - elapsed)
                    if "timeout" in kwargs:
                        kwargs["timeout"] = remaining
                    return await asyncio.to_thread(func, *args, **kwargs)
                finally:
                    self._semaphore.release()
            else:
                async with self._semaphore:
                    return await asyncio.to_thread(func, *args, **kwargs)
        return await asyncio.to_thread(func, *args, **kwargs)

    # ── JXL async ──

    async def encode_async(
        self,
        input,
        effort=None,
        distance=None,
        lossless=None,
        decoding_speed=None,
        *,
        exif=None,
        xmp=None,
        jumbf=None,
        icc=None,
        timeout=None,
    ):
        """Asynchronously encode a numpy array to JXL bytes."""

        def _worker(timeout=None):
            nonlocal input
            if hasattr(input, "flags") and not input.flags.c_contiguous:
                input = np.ascontiguousarray(input)
            return self.encode(
                input,
                effort=effort,
                distance=distance,
                lossless=lossless,
                decoding_speed=decoding_speed,
                exif=exif,
                xmp=xmp,
                jumbf=jumbf,
                icc=icc,
                timeout=timeout,
            )

        return await self._run_guarded(_worker, timeout=timeout)

    async def decode_async(self, data, *, metadata=False, out=None, timeout=None):
        """Asynchronously decode JXL bytes to a numpy array."""
        return await self._run_guarded(
            self.decode, data, metadata=metadata, out=out, timeout=timeout
        )

    async def read_async(
        self, path, *, metadata=False, out=None, use_mmap=False, timeout=None
    ):
        """Asynchronously read a JXL file and return a numpy array."""
        return await self._run_guarded(
            self.read,
            path,
            metadata=metadata,
            out=out,
            use_mmap=use_mmap,
            timeout=timeout,
        )

    async def write_async(
        self,
        path,
        image,
        effort=None,
        distance=None,
        lossless=None,
        decoding_speed=None,
        *,
        exif=None,
        xmp=None,
        jumbf=None,
        icc=None,
        timeout=None,
    ):
        """Asynchronously encode and write to a JXL file."""

        def _worker(timeout=None):
            nonlocal image
            if hasattr(image, "flags") and not image.flags.c_contiguous:
                image = np.ascontiguousarray(image)
            self.write(
                path,
                image,
                effort=effort,
                distance=distance,
                lossless=lossless,
                decoding_speed=decoding_speed,
                exif=exif,
                xmp=xmp,
                jumbf=jumbf,
                icc=icc,
                timeout=timeout,
            )

        return await self._run_guarded(_worker, timeout=timeout)

    # ── JPEG async ──

    async def encode_jpeg_async(self, input, quality=95):
        """Asynchronously encode numpy array to JPEG bytes."""

        def _worker():
            nonlocal input
            if hasattr(input, "flags") and not input.flags.c_contiguous:
                input = np.ascontiguousarray(input)
            return self.encode_jpeg(input, quality=quality)

        return await self._run_guarded(_worker)

    async def decode_jpeg_async(self, data, *, out=None):
        """Asynchronously decode JPEG bytes to numpy array."""
        return await self._run_guarded(self.decode_jpeg, data, out=out)

    async def read_jpeg_async(self, path, *, out=None, use_mmap=False):
        """Asynchronously read a JPEG file."""
        return await self._run_guarded(
            self.read_jpeg, path, out=out, use_mmap=use_mmap
        )

    async def write_jpeg_async(self, path, image, quality=95):
        """Asynchronously write a JPEG file."""

        def _worker():
            nonlocal image
            if hasattr(image, "flags") and not image.flags.c_contiguous:
                image = np.ascontiguousarray(image)
            self.write_jpeg(path, image, quality)

        return await self._run_guarded(_worker)

    # ── Cross-format async ──

    async def jpeg_to_jxl_async(self, data, effort=None, *, timeout=None):
        """Asynchronously transcode JPEG bytes to JXL bytes."""
        return await self._run_guarded(self.jpeg_to_jxl, data, effort, timeout=timeout)

    async def jxl_to_jpeg_async(self, data, *, timeout=None):
        """Asynchronously reconstruct JPEG bytes from JXL bytes."""
        return await self._run_guarded(self.jxl_to_jpeg, data, timeout=timeout)

    async def convert_jpeg_to_jxl_async(
        self, jpeg_path, jxl_path, effort=None, *, timeout=None
    ):
        """Asynchronously convert a JPEG file to JXL file."""
        return await self._run_guarded(
            self.convert_jpeg_to_jxl,
            jpeg_path,
            jxl_path,
            effort=effort,
            timeout=timeout,
        )

    async def convert_jxl_to_jpeg_async(self, jxl_path, jpeg_path, *, timeout=None):
        """Asynchronously convert a JXL file to JPEG file."""
        return await self._run_guarded(
            self.convert_jxl_to_jpeg, jxl_path, jpeg_path, timeout=timeout
        )

