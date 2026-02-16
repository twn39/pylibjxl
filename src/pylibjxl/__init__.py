import asyncio
from pathlib import Path

from ._pylibjxl import (  # type: ignore
    JXL as _JXL,
)
from ._pylibjxl import (  # type: ignore
    decode,
    decode_jpeg,
    decoder_version,
    encode,
    encode_jpeg,
    encoder_version,
    jpeg_to_jxl,
    jxl_to_jpeg,
    version,
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
    "read_jpeg",
    "write_jpeg",
    "read_jpeg_async",
    "write_jpeg_async",
    "jpeg_to_jxl",
    "jxl_to_jpeg",
    "encode_jpeg_async",
    "decode_jpeg_async",
    "jpeg_to_jxl_async",
    "jxl_to_jpeg_async",
    "convert_jpeg_to_jxl",
    "convert_jxl_to_jpeg",
    "convert_jpeg_to_jxl_async",
    "convert_jxl_to_jpeg_async",
]


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
):
    """
    Asynchronously encode a numpy array (H, W, C) to JXL bytes.
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
    )


async def decode_async(data, *, metadata=False):
    """
    Asynchronously decode JXL bytes to a numpy array (H, W, C).
    Releases the GIL during the decoding process.

    When metadata=True, returns (array, dict) with extracted metadata.
    """
    return await asyncio.to_thread(decode, data, metadata)


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
):
    """Encode a numpy array and write it to a JXL file.

    Args:
        path: Output file path (str or Path).
        image: uint8 numpy array of shape (height, width, channels).
        effort: Encoding effort [1-11] (default 7).
        distance: Perceptual distance [0.0-25.0] (default 1.0).
        lossless: If True, encode losslessly (default False).
        decoding_speed: Decoding speed tier [0-4] (default 0).
        exif: Optional EXIF metadata as bytes.
        xmp: Optional XMP metadata as bytes.
        jumbf: Optional JUMBF metadata as bytes.
    """
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = encode(
        image, effort, distance, lossless, decoding_speed, exif, xmp, jumbf
    )
    filepath.write_bytes(data)


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
    )


class JXL(_JXL):
    """Unified JXL/JPEG codec with synchronous context manager support.

    Owns a shared thread pool that is destroyed on close()/exit.
    Supports JXL encode/decode, JPEG encode/decode, cross-format
    transcoding, and file I/O for both formats.

    Usage::

        with pylibjxl.JXL(effort=7, threads=4) as jxl:
            # JXL
            jxl.write("output.jxl", image, exif=exif_bytes)
            img, meta = jxl.read("output.jxl", metadata=True)
            # JPEG
            jxl.write_jpeg("output.jpg", image, quality=95)
            img2 = jxl.read_jpeg("output.jpg")
            # Cross-format
            jxl.convert_jpeg_to_jxl("input.jpg", "output.jxl")
            jxl.convert_jxl_to_jpeg("input.jxl", "output.jpg")

    Args:
        effort: Encoding effort [1-11] (default 7).
        distance: Perceptual distance [0.0-25.0] (default 1.0).
        lossless: If True, encode losslessly (default False).
        decoding_speed: Decoding speed tier [0-4] (default 0).
        threads: Number of worker threads to use (default 0 = auto-detect).
                 For asyncio/FastAPI, set this to a fixed value (e.g., 4-8)
                 to prevent thread explosion.
    """

    # ── JXL File I/O ──

    def read(self, path, *, metadata=False):
        """Read a JXL file and return a numpy array."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"No such file: '{filepath}'")
        data = filepath.read_bytes()
        return self.decode(data, metadata)

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
    ):
        """Encode a numpy array and write it to a JXL file."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = self.encode(
            image, effort, distance, lossless, decoding_speed, exif, xmp, jumbf
        )
        filepath.write_bytes(data)

    # ── JPEG File I/O ──

    def read_jpeg(self, path):
        """Read a JPEG file and return a numpy array (H, W, 3)."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"No such file: '{filepath}'")
        data = filepath.read_bytes()
        return self.decode_jpeg(data)

    def write_jpeg(self, path, image, quality=95):
        """Encode a numpy array and write it to a JPEG file."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = self.encode_jpeg(image, quality=quality)
        filepath.write_bytes(data)

    # ── Cross-Format File Conversion ──

    def convert_jpeg_to_jxl(self, jpeg_path, jxl_path, effort=None):
        """Convert a JPEG file to JXL file (lossless transcoding)."""
        jpeg_filepath = Path(jpeg_path)
        if not jpeg_filepath.exists():
            raise FileNotFoundError(f"No such file: '{jpeg_filepath}'")
        jxl_filepath = Path(jxl_path)
        jxl_filepath.parent.mkdir(parents=True, exist_ok=True)
        jpeg_data = jpeg_filepath.read_bytes()
        jxl_data = self.jpeg_to_jxl(jpeg_data, effort=effort)
        jxl_filepath.write_bytes(jxl_data)

    def convert_jxl_to_jpeg(self, jxl_path, jpeg_path):
        """Convert a JXL file to JPEG file (lossless reconstruction)."""
        jxl_filepath = Path(jxl_path)
        if not jxl_filepath.exists():
            raise FileNotFoundError(f"No such file: '{jxl_filepath}'")
        jpeg_filepath = Path(jpeg_path)
        jpeg_filepath.parent.mkdir(parents=True, exist_ok=True)
        jxl_data = jxl_filepath.read_bytes()
        jpeg_data = self.jxl_to_jpeg(jxl_data)
        jpeg_filepath.write_bytes(jpeg_data)


class AsyncJXL(_JXL):
    """Unified JXL/JPEG codec with async context manager support.

    Owns a shared thread pool that is destroyed on close()/exit.

    Usage::

        async with pylibjxl.AsyncJXL(effort=7, threads=4) as jxl:
            await jxl.write_async("output.jxl", image)
            await jxl.write_jpeg_async("output.jpg", image)
            await jxl.convert_jpeg_to_jxl_async("in.jpg", "out.jxl")

    Args:
        effort: Encoding effort [1-11] (default 7).
        distance: Perceptual distance [0.0-25.0] (default 1.0).
        lossless: If True, encode losslessly (default False).
        decoding_speed: Decoding speed tier [0-4] (default 0).
        threads: Number of worker threads to use (default 0 = auto-detect).
                 For asyncio/FastAPI, set this to a fixed value (e.g., 4-8)
                 to prevent thread explosion.
    """

    async def __aenter__(self):
        self.__enter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__exit__(exc_type, exc_val, exc_tb)

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
    ):
        """Asynchronously encode a numpy array to JXL bytes."""
        return await asyncio.to_thread(
            self.encode,
            input,
            effort,
            distance,
            lossless,
            decoding_speed,
            exif,
            xmp,
            jumbf,
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
    ):
        """Asynchronously encode and write to a JXL file."""
        data = await asyncio.to_thread(
            self.encode,
            image,
            effort,
            distance,
            lossless,
            decoding_speed,
            exif,
            xmp,
            jumbf,
        )
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(filepath.write_bytes, data)

    # ── JPEG async ──

    async def encode_jpeg_async(self, input, quality=95):
        """Asynchronously encode numpy array to JPEG bytes."""
        return await asyncio.to_thread(self.encode_jpeg, input, quality)

    async def decode_jpeg_async(self, data):
        """Asynchronously decode JPEG bytes to numpy array."""
        return await asyncio.to_thread(self.decode_jpeg, data)

    async def read_jpeg_async(self, path):
        """Asynchronously read a JPEG file."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"No such file: '{filepath}'")
        data = await asyncio.to_thread(filepath.read_bytes)
        return await asyncio.to_thread(self.decode_jpeg, data)

    async def write_jpeg_async(self, path, image, quality=95):
        """Asynchronously write a JPEG file."""
        data = await asyncio.to_thread(self.encode_jpeg, image, quality)
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(filepath.write_bytes, data)

    # ── Cross-format async ──

    async def jpeg_to_jxl_async(self, data, effort=None):
        """Asynchronously transcode JPEG bytes to JXL bytes."""
        return await asyncio.to_thread(self.jpeg_to_jxl, data, effort)

    async def jxl_to_jpeg_async(self, data):
        """Asynchronously reconstruct JPEG bytes from JXL bytes."""
        return await asyncio.to_thread(self.jxl_to_jpeg, data)

    async def convert_jpeg_to_jxl_async(self, jpeg_path, jxl_path, effort=None):
        """Asynchronously convert a JPEG file to JXL file."""

        def _convert():
            jpeg_filepath = Path(jpeg_path)
            if not jpeg_filepath.exists():
                raise FileNotFoundError(f"No such file: '{jpeg_filepath}'")
            jxl_filepath = Path(jxl_path)
            jxl_filepath.parent.mkdir(parents=True, exist_ok=True)
            jpeg_data = jpeg_filepath.read_bytes()
            jxl_data = self.jpeg_to_jxl(jpeg_data, effort=effort)
            jxl_filepath.write_bytes(jxl_data)

        await asyncio.to_thread(_convert)

    async def convert_jxl_to_jpeg_async(self, jxl_path, jpeg_path):
        """Asynchronously convert a JXL file to JPEG file."""

        def _convert():
            jxl_filepath = Path(jxl_path)
            if not jxl_filepath.exists():
                raise FileNotFoundError(f"No such file: '{jxl_filepath}'")
            jpeg_filepath = Path(jpeg_path)
            jpeg_filepath.parent.mkdir(parents=True, exist_ok=True)
            jxl_data = jxl_filepath.read_bytes()
            jpeg_data = self.jxl_to_jpeg(jxl_data)
            jpeg_filepath.write_bytes(jpeg_data)

        await asyncio.to_thread(_convert)


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


def read_jpeg(path):
    """Read a JPEG image file and return a numpy array (H, W, 3).

    Args:
        path: Path to a .jpg/.jpeg file (str or Path).

    Returns:
        numpy.ndarray of shape (H, W, 3), dtype uint8.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"No such file: '{filepath}'")
    data = filepath.read_bytes()
    return decode_jpeg(data)


def write_jpeg(path, image, quality=95):
    """Encode a numpy array and write it to a JPEG file.

    Args:
        path: Output file path (str or Path).
        image: uint8 numpy array of shape (H, W, 3) or (H, W, 4).
        quality: JPEG quality [1-100] (default 95).
    """
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = encode_jpeg(image, quality=quality)
    filepath.write_bytes(data)


async def read_jpeg_async(path):
    """Asynchronously read a JPEG image file and return a numpy array."""
    return await asyncio.to_thread(read_jpeg, path)


async def write_jpeg_async(path, image, quality=95):
    """Asynchronously encode a numpy array and write it to a JPEG file."""
    return await asyncio.to_thread(write_jpeg, path, image, quality)


def convert_jpeg_to_jxl(jpeg_path, jxl_path, effort=7):
    """Convert a JPEG file to JXL file (lossless transcoding).

    The JPEG reconstruction data is preserved, so the original JPEG
    can be restored from the JXL file using convert_jxl_to_jpeg().

    Args:
        jpeg_path: Input JPEG file path (str or Path).
        jxl_path: Output JXL file path (str or Path).
        effort: Encoding effort [1-10] (default 7).
    """
    jpeg_filepath = Path(jpeg_path)
    if not jpeg_filepath.exists():
        raise FileNotFoundError(f"No such file: '{jpeg_filepath}'")
    jxl_filepath = Path(jxl_path)
    jxl_filepath.parent.mkdir(parents=True, exist_ok=True)
    jpeg_data = jpeg_filepath.read_bytes()
    jxl_data = jpeg_to_jxl(jpeg_data, effort=effort)
    jxl_filepath.write_bytes(jxl_data)


def convert_jxl_to_jpeg(jxl_path, jpeg_path):
    """Convert a JXL file to JPEG file.

    If the JXL was created via lossless JPEG transcoding (jpeg_to_jxl),
    the original JPEG is reconstructed losslessly. Otherwise raises an error.

    Args:
        jxl_path: Input JXL file path (str or Path).
        jpeg_path: Output JPEG file path (str or Path).
    """
    jxl_filepath = Path(jxl_path)
    if not jxl_filepath.exists():
        raise FileNotFoundError(f"No such file: '{jxl_filepath}'")
    jpeg_filepath = Path(jpeg_path)
    jpeg_filepath.parent.mkdir(parents=True, exist_ok=True)
    jxl_data = jxl_filepath.read_bytes()
    jpeg_data = jxl_to_jpeg(jxl_data)
    jpeg_filepath.write_bytes(jpeg_data)


async def convert_jpeg_to_jxl_async(jpeg_path, jxl_path, effort=7):
    """Async convert a JPEG file to JXL file (lossless transcoding)."""
    return await asyncio.to_thread(convert_jpeg_to_jxl, jpeg_path, jxl_path, effort)


async def convert_jxl_to_jpeg_async(jxl_path, jpeg_path):
    """Async convert a JXL file to JPEG file."""
    return await asyncio.to_thread(convert_jxl_to_jpeg, jxl_path, jpeg_path)
