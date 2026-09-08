"""Pillow ImagePlugin for JPEG XL (JXL) powered by pylibjxl.

Provides seamless, high-performance, transparent Image.open() and Image.save()
support for JPEG XL images with lazy loading, full metadata support (EXIF, ICC, XMP),
and thread pool integration.
"""

from typing import Any, Union

import numpy as np

try:
    from PIL import Image, ImageFile
except ImportError:
    Image = None  # type: ignore[assignment]
    ImageFile = None  # type: ignore[assignment]

import pylibjxl

_JXL_HEADER_PREFIXES = (
    b"\xff\x0a",  # Naked JXL codestream
    b"\x00\x00\x00\x0c\x4a\x58\x4c\x20\x0d\x0a\x87\x0a",  # ISOBMFF container
)


def _accept(prefix: bytes) -> bool:
    """Fast signature check for JXL codestream or container format."""
    if len(prefix) < 2:
        return False
    if prefix[:2] == b"\xff\x0a":
        return True
    if (
        len(prefix) >= 12
        and prefix[:12] == b"\x00\x00\x00\x0c\x4a\x58\x4c\x20\x0d\x0a\x87\x0a"
    ):
        return True
    if len(prefix) >= 8 and prefix[4:8] == b"JXL ":
        return True
    return False


if ImageFile is not None:

    class JxlImageFile(ImageFile.ImageFile):
        """Pillow ImageFile plugin for JPEG XL images with true deferred decoding."""

        format = "JXL"
        format_description = "JPEG XL image"

        def _open(self) -> None:
            # 1. Inspect header without decoding pixels (Lazy Loading)
            # Read enough initial bytes to probe header metadata
            assert self.fp is not None
            pos = self.fp.tell() if hasattr(self.fp, "tell") else 0
            # Read first 64KB for probing
            header_bytes = self.fp.read(65536)
            if hasattr(self.fp, "seek"):
                self.fp.seek(pos)

            try:
                info = pylibjxl.probe(header_bytes)
            except Exception:
                # If 64KB wasn't enough (rare large header boxes), read full buffer
                if hasattr(self.fp, "seek"):
                    self.fp.seek(pos)
                full_data = self.fp.read()
                if hasattr(self.fp, "seek"):
                    self.fp.seek(pos)
                info = pylibjxl.probe(full_data)

            self._size = (info["width"], info["height"])
            channels = info["channels"]
            has_alpha = info["has_alpha"]

            if channels == 1:
                self._mode = "L"
                self.rawmode = "L"
            elif channels == 2 or (channels == 1 and has_alpha):
                self._mode = "LA"
                self.rawmode = "LA"
            elif channels == 4 or (channels == 3 and has_alpha):
                self._mode = "RGBA"
                self.rawmode = "RGBA"
            else:
                self._mode = "RGB"
                self.rawmode = "RGB"

            self.info["bits_per_sample"] = info.get("bits_per_sample", 8)
            self.info["suggested_threads"] = info.get("suggested_threads", 1)
            self.is_animated = info.get("have_animation", False)
            self.tile = [("jxl", (0, 0) + self.size, 0, None)]  # type: ignore[assignment]

        def load(self):
            """Decode pixel data when requested by Pillow or user."""
            if self.tile:
                if self.fp is not None:
                    if hasattr(self.fp, "seek"):
                        self.fp.seek(0)
                    data = self.fp.read()

                    # Close exclusive fp if owned
                    if self._exclusive_fp and self.fp:
                        self.fp.close()
                    self.fp = None

                    # Decode with full metadata
                    array, meta = pylibjxl.decode(data, metadata=True)

                    # Populate metadata
                    if "icc" in meta:
                        self.info["icc_profile"] = meta["icc"]
                    if "exif" in meta:
                        self.info["exif"] = meta["exif"]
                    if "xmp" in meta:
                        self.info["xmp"] = meta["xmp"]
                    if "jumbf" in meta:
                        self.info["jumbf"] = meta["jumbf"]

                    # Directly load pixel buffer into PIL Image
                    target_mode = self.mode
                    if array.ndim == 2 and target_mode != "L":
                        target_mode = "L"
                        self._mode = "L"
                    elif array.ndim == 3:
                        c = array.shape[2]
                        if c == 1 and target_mode != "L":
                            target_mode = "L"
                            self._mode = "L"
                            array = array.squeeze(-1)
                        elif c == 3 and target_mode != "RGB":
                            target_mode = "RGB"
                            self._mode = "RGB"
                        elif c == 4 and target_mode != "RGBA":
                            target_mode = "RGBA"
                            self._mode = "RGBA"

                    if not array.flags.c_contiguous:
                        array = np.ascontiguousarray(array)

                    self._im = Image.core.new(target_mode, self.size)
                    self.frombytes(array.tobytes())
                self.tile = []

            return Image.Image.load(self)


else:
    JxlImageFile = None


def _save(
    im: Any, fp: Any, filename: Union[str, bytes], save_all: bool = False
) -> None:
    """Save driver for writing JXL files from a Pillow Image."""
    if Image is None:
        raise ImportError("Pillow must be installed to use _save")

    info = im.encoderinfo.copy()

    # Determine lossless / distance / quality
    lossless = info.get("lossless", False)
    quality = info.get("quality")
    distance = info.get("distance")

    if distance is None:
        if lossless or (quality is not None and quality >= 100):
            distance = 0.0
            lossless = True
        elif quality is not None:
            # Map quality [1, 100] to Butteraugli distance [15.0, 0.1]
            q = max(1, min(100, int(quality)))
            if q >= 100:
                distance = 0.0
                lossless = True
            elif q >= 30:
                distance = 0.1 + (100 - q) * 0.09
            else:
                distance = 6.4 + (30 - q) * 0.25
        else:
            distance = 1.0  # Default visually lossless

    effort = info.get("effort", 7)
    decoding_speed = info.get("decoding_speed", 0)

    # Extract metadata
    exif = info.get("exif")
    if exif is None:
        exif = im.info.get("exif")
    if exif is None:
        try:
            exif_obj = im.getexif()
            if exif_obj:
                exif = exif_obj.tobytes()
        except Exception:
            exif = None

    if hasattr(exif, "tobytes"):
        exif = exif.tobytes()

    icc = info.get("icc_profile") or im.info.get("icc_profile")
    xmp = info.get("xmp") or im.info.get("xmp")
    jumbf = info.get("jumbf") or im.info.get("jumbf")

    # Format conversion
    if im.mode not in ("RGB", "RGBA", "L", "LA"):
        if "A" in im.mode:
            im = im.convert("RGBA")
        elif im.mode == "P":
            im = im.convert("RGBA" if "transparency" in im.info else "RGB")
        else:
            im = im.convert("RGB")

    # Convert to contiguous uint8 ndarray
    arr = np.asarray(im)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)

    jxl_bytes = pylibjxl.encode(
        arr,
        effort=effort,
        distance=distance,
        lossless=lossless,
        decoding_speed=decoding_speed,
        exif=exif,
        xmp=xmp,
        jumbf=jumbf,
        icc=icc,
    )

    fp.write(jxl_bytes)


_registered = False


def register_pillow(override: bool = True) -> None:
    """Register pylibjxl transparently as Pillow's official JXL codec plugin.

    Enables `Image.open()` and `im.save()` with `.jxl` format support.

    Args:
        override: If True, overrides any existing JXL plugins (e.g. pillow-jxl-plugin).
                  Defaults to True.
    """
    global _registered
    if Image is None or ImageFile is None:
        raise ImportError(
            "Pillow is not installed. Install it with `pip install Pillow` to use register_pillow()."
        )

    format_id = "JXL"

    if override or format_id not in Image.ID:
        Image.register_open(format_id, JxlImageFile, _accept)
        Image.register_save(format_id, _save)
        Image.register_extension(format_id, ".jxl")
        Image.register_mime(format_id, "image/jxl")
        _registered = True
