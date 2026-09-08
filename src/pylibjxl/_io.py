from pathlib import Path

from ._pylibjxl import (  # type: ignore
    decode,
    decode_jpeg,
    encode,
    encode_jpeg,
    jpeg_to_jxl,
    jxl_to_jpeg,
)


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
    icc=None,
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
    """
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = encode(
        image, effort, distance, lossless, decoding_speed, exif, xmp, jumbf, icc
    )
    filepath.write_bytes(data)


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
        jpeg_path: Input JPEG file path (str or Path).
    """
    jxl_filepath = Path(jxl_path)
    if not jxl_filepath.exists():
        raise FileNotFoundError(f"No such file: '{jxl_filepath}'")
    jpeg_filepath = Path(jpeg_path)
    jpeg_filepath.parent.mkdir(parents=True, exist_ok=True)
    jxl_data = jxl_filepath.read_bytes()
    jpeg_data = jxl_to_jpeg(jxl_data)
    jpeg_filepath.write_bytes(jpeg_data)
