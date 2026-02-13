import asyncio
import os
from pathlib import Path

from ._pylibjxl import (
    version,
    decoder_version,
    encoder_version,
    encode,
    decode,
    JXL as _JXL,
    encode_jpeg,
    decode_jpeg,
    jpeg_to_jxl,
    jxl_to_jpeg,
)

__all__ = [
    "version",
    "decoder_version",
    "encoder_version",
    "encode",
    "decode",
    "encode_async",
    "decode_async",
    "read",
    "write",
    "read_async",
    "write_async",
    "JXL",
    "AsyncJXL",
    "encode_jpeg",
    "decode_jpeg",
    "jpeg_to_jxl",
    "jxl_to_jpeg",
    "encode_jpeg_async",
    "decode_jpeg_async",
    "jpeg_to_jxl_async",
    "jxl_to_jpeg_async",
]


# ─── Free Functions: In-Memory ──────────────────────────────────────────────────


async def encode_async(
    input, effort=7, distance=1.0, lossless=False, *, exif=None, xmp=None, jumbf=None
):
    """
    Asynchronously encode a numpy array (H, W, C) to JXL bytes.
    Releases the GIL during the encoding process.
    """
    return await asyncio.to_thread(
        encode, input, effort, distance, lossless, exif, xmp, jumbf
    )


async def decode_async(data, *, metadata=False):
    """
    Asynchronously decode JXL bytes to a numpy array (H, W, C).
    Releases the GIL during the decoding process.

    When metadata=True, returns (array, dict) with extracted metadata.
    """
    return await asyncio.to_thread(decode, data, metadata)


# ─── Free Functions: File I/O ───────────────────────────────────────────────────


def read(path, *, metadata=False):
    """Read a JXL image file and return a numpy array (H, W, C).

    Args:
        path: Path to a .jxl file (str or Path).
        metadata: If True, also return metadata dict (default False).

    Returns:
        numpy.ndarray when metadata=False,
        tuple(numpy.ndarray, dict) when metadata=True.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"No such file: '{filepath}'")
    data = filepath.read_bytes()
    return decode(data, metadata)


def write(
    path, image, effort=7, distance=1.0, lossless=False, *, exif=None, xmp=None, jumbf=None
):
    """Encode a numpy array and write it to a JXL file.

    Args:
        path: Output file path (str or Path).
        image: uint8 numpy array of shape (height, width, channels).
        effort: Encoding effort [1-10] (default 7).
        distance: Perceptual distance [0.0-25.0] (default 1.0).
        lossless: If True, encode losslessly (default False).
        exif: Optional EXIF metadata as bytes.
        xmp: Optional XMP metadata as bytes.
        jumbf: Optional JUMBF metadata as bytes.
    """
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = encode(image, effort, distance, lossless, exif, xmp, jumbf)
    filepath.write_bytes(data)


async def read_async(path, *, metadata=False):
    """Asynchronously read a JXL image file and return a numpy array."""
    return await asyncio.to_thread(read, path, metadata=metadata)


async def write_async(
    path, image, effort=7, distance=1.0, lossless=False, *, exif=None, xmp=None, jumbf=None
):
    """Asynchronously encode a numpy array and write it to a JXL file."""
    return await asyncio.to_thread(
        write, path, image, effort, distance, lossless, exif=exif, xmp=xmp, jumbf=jumbf
    )


# ─── Context Manager: Sync ──────────────────────────────────────────────────────


class JXL(_JXL):
    """Unified JXL codec with synchronous context manager support.

    Usage::

        with pylibjxl.JXL(effort=7) as jxl:
            jxl.write("output.jxl", image, exif=exif_bytes)
            img, meta = jxl.read("output.jxl", metadata=True)
            data = jxl.encode(image, xmp=xmp_bytes)
            img2 = jxl.decode(data)
    """

    def read(self, path, *, metadata=False):
        """Read a JXL file and return a numpy array.

        Args:
            path: Path to a .jxl file (str or Path).
            metadata: If True, also return metadata dict.

        Returns:
            numpy.ndarray or tuple(numpy.ndarray, dict).
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"No such file: '{filepath}'")
        data = filepath.read_bytes()
        return self.decode(data, metadata)

    def write(self, path, image, effort=None, distance=None, lossless=None,
              *, exif=None, xmp=None, jumbf=None):
        """Encode a numpy array and write it to a JXL file.

        Per-call overrides take precedence over constructor defaults.

        Args:
            path: Output file path (str or Path).
            image: uint8 numpy array of shape (height, width, channels).
            effort: Encoding effort [1-10] (optional, uses default).
            distance: Perceptual distance [0.0-25.0] (optional, uses default).
            lossless: If True, encode losslessly (optional, uses default).
            exif: Optional EXIF metadata as bytes.
            xmp: Optional XMP metadata as bytes.
            jumbf: Optional JUMBF metadata as bytes.
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = self.encode(image, effort, distance, lossless, exif, xmp, jumbf)
        filepath.write_bytes(data)


# ─── Context Manager: Async ─────────────────────────────────────────────────────


class AsyncJXL(_JXL):
    """Unified JXL codec with async context manager support.

    Usage::

        async with pylibjxl.AsyncJXL(effort=7) as jxl:
            await jxl.write_async("output.jxl", image, exif=exif_bytes)
            img, meta = await jxl.read_async("output.jxl", metadata=True)
    """

    async def __aenter__(self):
        self.__enter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__exit__(exc_type, exc_val, exc_tb)

    async def encode_async(self, input, effort=None, distance=None, lossless=None,
                           *, exif=None, xmp=None, jumbf=None):
        """Asynchronously encode a numpy array to JXL bytes."""
        return await asyncio.to_thread(
            self.encode, input, effort, distance, lossless, exif, xmp, jumbf
        )

    async def decode_async(self, data, *, metadata=False):
        """Asynchronously decode JXL bytes to a numpy array."""
        return await asyncio.to_thread(self.decode, data, metadata)

    async def read_async(self, path, *, metadata=False):
        """Asynchronously read a JXL file and return a numpy array."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"No such file: '{filepath}'")
        data = await asyncio.to_thread(filepath.read_bytes)
        return await asyncio.to_thread(self.decode, data, metadata)

    async def write_async(self, path, image, effort=None, distance=None, lossless=None):
        """Asynchronously encode and write to a JXL file."""
        data = await asyncio.to_thread(self.encode, image, effort, distance, lossless)
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(filepath.write_bytes, data)


# ─── JPEG & Transcoding ───

async def encode_jpeg_async(input, quality=95):
    """Async encode numpy array to JPEG bytes."""
    return await asyncio.to_thread(encode_jpeg, input, quality=quality)

async def decode_jpeg_async(data):
    """Async decode JPEG bytes to numpy array."""
    return await asyncio.to_thread(decode_jpeg, data)

async def jpeg_to_jxl_async(data, effort=7):
    """Async losslessly recompress JPEG bytes to JXL bytes."""
    return await asyncio.to_thread(jpeg_to_jxl, data, effort=effort)

async def jxl_to_jpeg_async(data):
    """Async reconstruct original JPEG bytes from JXL bytes."""
    return await asyncio.to_thread(jxl_to_jpeg, data)
