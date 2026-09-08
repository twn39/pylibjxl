import mmap
from pathlib import Path

import numpy as np

from ._pylibjxl import (  # type: ignore
    decode,
    decode_jpeg,
    jpeg_to_jxl_file,
    jxl_to_jpeg_file,
)
from ._pylibjxl import (  # type: ignore
    encode as _encode,
)
from ._pylibjxl import (  # type: ignore
    encode_jpeg as _encode_jpeg,
)


def encode(
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
    """Encode a numpy array (H, W, C) to JXL bytes.

    Automatically handles non-contiguous arrays safely.
    """
    if hasattr(input, "flags") and not input.flags.c_contiguous:
        input = np.ascontiguousarray(input)
    return _encode(
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


def encode_jpeg(input, quality=95):
    """Encode a numpy array (H, W, 3/4) to JPEG bytes using libjpeg-turbo.

    Automatically handles non-contiguous arrays safely.
    """
    if hasattr(input, "flags") and not input.flags.c_contiguous:
        input = np.ascontiguousarray(input)
    return _encode_jpeg(input, quality=quality)


def read(path, *, metadata=False, out=None, use_mmap=False, timeout=None):
    """Read a JXL image file and return a numpy array (H, W, C).

    Args:
        path: Path to a .jxl file (str or Path).
        metadata: If True, also return metadata dict (default False).
        out: Optional pre-allocated C-contiguous uint8 numpy array for zero-copy in-place decode.
        use_mmap: If True, uses memory-mapped file for zero-copy reading (default False).
        timeout: Optional acquisition timeout in seconds.

    Returns:
        numpy.ndarray when metadata=False,
        tuple(numpy.ndarray, dict) when metadata=True.

    Raises:
        FileNotFoundError: If the file does not exist.
        TimeoutError: If acquiring a runner exceeds timeout.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"No such file: '{filepath}'")

    if use_mmap:
        with open(filepath, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return decode(mm, metadata=metadata, out=out, timeout=timeout)
    else:
        data = filepath.read_bytes()
        return decode(data, metadata=metadata, out=out, timeout=timeout)


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
    icc=None,
    timeout=None,
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
        icc: Optional ICC profile metadata as bytes.
        timeout: Optional acquisition timeout in seconds.
    """
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = encode(
        image,
        effort,
        distance,
        lossless,
        decoding_speed,
        exif=exif,
        xmp=xmp,
        jumbf=jumbf,
        icc=icc,
        timeout=timeout,
    )
    filepath.write_bytes(data)


def read_jpeg(path, *, out=None, use_mmap=False):
    """Read a JPEG image file and return a numpy array (H, W, 3).

    Args:
        path: Path to a .jpg/.jpeg file (str or Path).
        out: Optional pre-allocated C-contiguous uint8 numpy array for in-place decode.
        use_mmap: If True, uses memory-mapped file for zero-copy reading (default False).

    Returns:
        numpy.ndarray of shape (H, W, 3), dtype uint8.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"No such file: '{filepath}'")

    if use_mmap:
        with open(filepath, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return decode_jpeg(mm, out=out)
    else:
        data = filepath.read_bytes()
        return decode_jpeg(data, out=out)


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


def convert_jpeg_to_jxl(jpeg_path, jxl_path, effort=7, *, timeout=None):
    """Convert a JPEG file to JXL file (lossless transcoding).

    The JPEG reconstruction data is preserved, so the original JPEG
    can be restored from the JXL file using convert_jxl_to_jpeg().

    Args:
        jpeg_path: Input JPEG file path (str or Path).
        jxl_path: Output JXL file path (str or Path).
        effort: Encoding effort [1-11] (default 7).
        timeout: Optional acquisition timeout in seconds.
    """
    jpeg_filepath = Path(jpeg_path)
    if not jpeg_filepath.exists():
        raise FileNotFoundError(f"No such file: '{jpeg_filepath}'")
    jxl_filepath = Path(jxl_path)
    jxl_filepath.parent.mkdir(parents=True, exist_ok=True)
    jpeg_to_jxl_file(str(jpeg_filepath), str(jxl_filepath), effort=effort, timeout=timeout)


def convert_jxl_to_jpeg(jxl_path, jpeg_path, *, timeout=None):
    """Convert a JXL file to JPEG file (lossless reconstruction).

    If the JXL was created via lossless JPEG transcoding (jpeg_to_jxl),
    the original JPEG is reconstructed losslessly. Otherwise raises an error.

    Args:
        jxl_path: Input JXL file path (str or Path).
        jpeg_path: Output JPEG file path (str or Path).
        timeout: Optional acquisition timeout in seconds.
    """
    jxl_filepath = Path(jxl_path)
    if not jxl_filepath.exists():
        raise FileNotFoundError(f"No such file: '{jxl_filepath}'")
    jpeg_filepath = Path(jpeg_path)
    jpeg_filepath.parent.mkdir(parents=True, exist_ok=True)
    jxl_to_jpeg_file(str(jxl_filepath), str(jpeg_filepath), timeout=timeout)
