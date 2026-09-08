import asyncio

import numpy as np

from ._io import (
    convert_jpeg_to_jxl,
    convert_jxl_to_jpeg,
    probe,
    probe_file,
    read,
    read_jpeg,
    write,
    write_jpeg,
)
from ._pylibjxl import (  # type: ignore
    decode,
    decode_jpeg,
    encode,
    encode_jpeg,
    jpeg_to_jxl,
    jxl_to_jpeg,
)


async def encode_async(
    input,
    effort=7,
    distance=1.0,
    lossless=False,
    decoding_speed=0,
    *,
    exif=None,
    xmp=None,
    jumbf=None,
    icc=None,
    timeout=None,
):
    """Asynchronously encode a numpy array (H, W, C) to JXL bytes.

    Releases the GIL during the encoding process.
    """

    def _worker():
        nonlocal input
        if hasattr(input, "flags") and not input.flags.c_contiguous:
            input = np.ascontiguousarray(input)
        return encode(
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

    return await asyncio.to_thread(_worker)


async def decode_async(data, *, metadata=False, out=None, timeout=None):
    """Asynchronously decode JXL bytes to a numpy array (H, W, C).

    Releases the GIL during the decoding process.

    When metadata=True, returns (array, dict) with extracted metadata.
    When out is provided, decodes in-place into the pre-allocated array.
    """
    return await asyncio.to_thread(
        decode, data, metadata=metadata, out=out, timeout=timeout
    )


async def read_async(path, *, metadata=False, out=None, use_mmap=False, timeout=None):
    """Asynchronously read a JXL image file and return a numpy array."""
    return await asyncio.to_thread(
        read, path, metadata=metadata, out=out, use_mmap=use_mmap, timeout=timeout
    )


async def probe_async(data):
    """Asynchronously probe JXL header metadata with zero pool contention."""
    return await asyncio.to_thread(probe, data)


async def probe_file_async(path, *, use_mmap=False):
    """Asynchronously probe JXL image file header metadata without decoding pixels."""
    return await asyncio.to_thread(probe_file, path, use_mmap=use_mmap)


async def write_async(
    path,
    image,
    effort=7,
    distance=1.0,
    lossless=False,
    decoding_speed=0,
    *,
    exif=None,
    xmp=None,
    jumbf=None,
    icc=None,
    timeout=None,
):
    """Asynchronously encode a numpy array and write it to a JXL file."""

    def _worker():
        nonlocal image
        if hasattr(image, "flags") and not image.flags.c_contiguous:
            image = np.ascontiguousarray(image)
        write(
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

    return await asyncio.to_thread(_worker)


async def encode_jpeg_async(input, quality=95):
    """Async encode numpy array to JPEG bytes."""

    def _worker():
        nonlocal input
        if hasattr(input, "flags") and not input.flags.c_contiguous:
            input = np.ascontiguousarray(input)
        return encode_jpeg(input, quality=quality)

    return await asyncio.to_thread(_worker)


async def decode_jpeg_async(data, *, out=None):
    """Async decode JPEG bytes to numpy array."""
    return await asyncio.to_thread(decode_jpeg, data, out=out)


async def jpeg_to_jxl_async(data, effort=7, *, timeout=None):
    """Async losslessly recompress JPEG bytes to JXL bytes."""
    return await asyncio.to_thread(jpeg_to_jxl, data, effort=effort, timeout=timeout)


async def jxl_to_jpeg_async(data, *, timeout=None):
    """Async reconstruct original JPEG bytes from JXL bytes."""
    return await asyncio.to_thread(jxl_to_jpeg, data, timeout=timeout)


async def read_jpeg_async(path, *, out=None, use_mmap=False):
    """Asynchronously read a JPEG image file and return a numpy array."""
    return await asyncio.to_thread(read_jpeg, path, out=out, use_mmap=use_mmap)


async def write_jpeg_async(path, image, quality=95):
    """Asynchronously encode a numpy array and write it to a JPEG file."""

    def _worker():
        nonlocal image
        if hasattr(image, "flags") and not image.flags.c_contiguous:
            image = np.ascontiguousarray(image)
        write_jpeg(path, image, quality)

    return await asyncio.to_thread(_worker)


async def convert_jpeg_to_jxl_async(jpeg_path, jxl_path, effort=7, *, timeout=None):
    """Async convert a JPEG file to JXL file (lossless transcoding)."""
    return await asyncio.to_thread(
        convert_jpeg_to_jxl, jpeg_path, jxl_path, effort=effort, timeout=timeout
    )


async def convert_jxl_to_jpeg_async(jxl_path, jpeg_path, *, timeout=None):
    """Async convert a JXL file to JPEG file."""
    return await asyncio.to_thread(
        convert_jxl_to_jpeg, jxl_path, jpeg_path, timeout=timeout
    )
