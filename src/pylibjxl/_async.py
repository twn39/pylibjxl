import asyncio

from ._io import (
    convert_jpeg_to_jxl,
    convert_jxl_to_jpeg,
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
):
    """Asynchronously encode a numpy array (H, W, C) to JXL bytes.

    Releases the GIL during the encoding process.
    """
    return await asyncio.to_thread(
        encode,
        input,
        effort,
        distance,
        lossless,
        decoding_speed,
        exif,
        xmp,
        jumbf,
        icc,
    )


async def decode_async(data, *, metadata=False):
    """Asynchronously decode JXL bytes to a numpy array (H, W, C).

    Releases the GIL during the decoding process.

    When metadata=True, returns (array, dict) with extracted metadata.
    """
    return await asyncio.to_thread(decode, data, metadata)


async def read_async(path, *, metadata=False):
    """Asynchronously read a JXL image file and return a numpy array."""
    return await asyncio.to_thread(read, path, metadata=metadata)


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
):
    """Asynchronously encode a numpy array and write it to a JXL file."""
    return await asyncio.to_thread(
        write,
        path,
        image,
        effort,
        distance,
        lossless,
        decoding_speed,
        exif=exif,
        xmp=xmp,
        jumbf=jumbf,
        icc=icc,
    )


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


async def read_jpeg_async(path):
    """Asynchronously read a JPEG image file and return a numpy array."""
    return await asyncio.to_thread(read_jpeg, path)


async def write_jpeg_async(path, image, quality=95):
    """Asynchronously encode a numpy array and write it to a JPEG file."""
    return await asyncio.to_thread(write_jpeg, path, image, quality)


async def convert_jpeg_to_jxl_async(jpeg_path, jxl_path, effort=7):
    """Async convert a JPEG file to JXL file (lossless transcoding)."""
    return await asyncio.to_thread(convert_jpeg_to_jxl, jpeg_path, jxl_path, effort)


async def convert_jxl_to_jpeg_async(jxl_path, jpeg_path):
    """Async convert a JXL file to JPEG file."""
    return await asyncio.to_thread(convert_jxl_to_jpeg, jxl_path, jpeg_path)
