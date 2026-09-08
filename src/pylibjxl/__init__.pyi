from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union, overload

import numpy as np
import numpy.typing as npt

BufferType = Union[bytes, bytearray, memoryview, npt.NDArray[np.uint8], Any]

# --- Native extension functions ---

def version() -> Dict[str, int]: ...
def decoder_version() -> int: ...
def encoder_version() -> int: ...
def encode(
    input: npt.NDArray[np.uint8],
    effort: int = 7,
    distance: float = 1.0,
    lossless: bool = False,
    decoding_speed: int = 0,
    exif: Optional[bytes] = None,
    xmp: Optional[bytes] = None,
    jumbf: Optional[bytes] = None,
    icc: Optional[bytes] = None,
) -> bytes: ...
@overload
def decode(
    data: BufferType,
    metadata: Literal[False] = False,
    out: Optional[npt.NDArray[np.uint8]] = None,
) -> npt.NDArray[np.uint8]: ...
@overload
def decode(
    data: BufferType,
    metadata: Literal[True],
    out: Optional[npt.NDArray[np.uint8]] = None,
) -> Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]: ...
@overload
def decode(
    data: BufferType, metadata: bool, out: Optional[npt.NDArray[np.uint8]] = None
) -> Union[npt.NDArray[np.uint8], Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]]: ...
def encode_jpeg(input: npt.NDArray[np.uint8], quality: int = 95) -> bytes: ...
def decode_jpeg(
    data: BufferType, out: Optional[npt.NDArray[np.uint8]] = None
) -> npt.NDArray[np.uint8]: ...
def jpeg_to_jxl(data: BufferType, effort: int = 7) -> bytes: ...
def jxl_to_jpeg(data: BufferType) -> bytes: ...
def jpeg_to_jxl_file(
    in_path: Union[str, Path], out_path: Union[str, Path], effort: int = 7
) -> None: ...
def jxl_to_jpeg_file(in_path: Union[str, Path], out_path: Union[str, Path]) -> None: ...

class _JXL:
    def __init__(
        self,
        effort: int = 7,
        distance: float = 1.0,
        lossless: bool = False,
        decoding_speed: int = 0,
        threads: int = 0,
    ) -> None: ...
    def encode(
        self,
        input: npt.NDArray[np.uint8],
        effort: Optional[int] = None,
        distance: Optional[float] = None,
        lossless: Optional[bool] = None,
        decoding_speed: Optional[int] = None,
        exif: Optional[bytes] = None,
        xmp: Optional[bytes] = None,
        jumbf: Optional[bytes] = None,
        icc: Optional[bytes] = None,
    ) -> bytes: ...
    @overload
    def decode(
        self,
        data: BufferType,
        metadata: Literal[False] = False,
        out: Optional[npt.NDArray[np.uint8]] = None,
    ) -> npt.NDArray[np.uint8]: ...
    @overload
    def decode(
        self,
        data: BufferType,
        metadata: Literal[True],
        out: Optional[npt.NDArray[np.uint8]] = None,
    ) -> Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]: ...
    @overload
    def decode(
        self,
        data: BufferType,
        metadata: bool,
        out: Optional[npt.NDArray[np.uint8]] = None,
    ) -> Union[
        npt.NDArray[np.uint8], Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]
    ]: ...
    def encode_jpeg(self, input: npt.NDArray[np.uint8], quality: int = 95) -> bytes: ...
    def decode_jpeg(
        self, data: BufferType, out: Optional[npt.NDArray[np.uint8]] = None
    ) -> npt.NDArray[np.uint8]: ...
    def jpeg_to_jxl(self, data: BufferType, effort: Optional[int] = None) -> bytes: ...
    def jxl_to_jpeg(self, data: BufferType) -> bytes: ...
    def jpeg_to_jxl_file(
        self,
        in_path: Union[str, Path],
        out_path: Union[str, Path],
        effort: Optional[int] = None,
    ) -> None: ...
    def jxl_to_jpeg_file(
        self, in_path: Union[str, Path], out_path: Union[str, Path]
    ) -> None: ...
    def close(self) -> None: ...
    @property
    def closed(self) -> bool: ...
    def __enter__(self) -> "_JXL": ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

# --- Python wrapper functions and classes ---

async def encode_async(
    input: npt.NDArray[np.uint8],
    effort: int = 7,
    distance: float = 1.0,
    lossless: bool = False,
    decoding_speed: int = 0,
    *,
    exif: Optional[bytes] = None,
    xmp: Optional[bytes] = None,
    jumbf: Optional[bytes] = None,
    icc: Optional[bytes] = None,
) -> bytes: ...
@overload
async def decode_async(
    data: BufferType,
    *,
    metadata: Literal[False] = False,
    out: Optional[npt.NDArray[np.uint8]] = None,
) -> npt.NDArray[np.uint8]: ...
@overload
async def decode_async(
    data: BufferType,
    *,
    metadata: Literal[True],
    out: Optional[npt.NDArray[np.uint8]] = None,
) -> Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]: ...
@overload
async def decode_async(
    data: BufferType, *, metadata: bool, out: Optional[npt.NDArray[np.uint8]] = None
) -> Union[npt.NDArray[np.uint8], Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]]: ...
@overload
def read(
    path: Union[str, Path],
    *,
    metadata: Literal[False] = False,
    out: Optional[npt.NDArray[np.uint8]] = None,
    use_mmap: bool = False,
) -> npt.NDArray[np.uint8]: ...
@overload
def read(
    path: Union[str, Path],
    *,
    metadata: Literal[True],
    out: Optional[npt.NDArray[np.uint8]] = None,
    use_mmap: bool = False,
) -> Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]: ...
@overload
def read(
    path: Union[str, Path],
    *,
    metadata: bool,
    out: Optional[npt.NDArray[np.uint8]] = None,
    use_mmap: bool = False,
) -> Union[npt.NDArray[np.uint8], Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]]: ...
def write(
    path: Union[str, Path],
    image: npt.NDArray[np.uint8],
    effort: int = 7,
    distance: float = 1.0,
    lossless: bool = False,
    decoding_speed: int = 0,
    *,
    exif: Optional[bytes] = None,
    xmp: Optional[bytes] = None,
    jumbf: Optional[bytes] = None,
    icc: Optional[bytes] = None,
) -> None: ...
@overload
async def read_async(
    path: Union[str, Path],
    *,
    metadata: Literal[False] = False,
    out: Optional[npt.NDArray[np.uint8]] = None,
    use_mmap: bool = False,
) -> npt.NDArray[np.uint8]: ...
@overload
async def read_async(
    path: Union[str, Path],
    *,
    metadata: Literal[True],
    out: Optional[npt.NDArray[np.uint8]] = None,
    use_mmap: bool = False,
) -> Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]: ...
@overload
async def read_async(
    path: Union[str, Path],
    *,
    metadata: bool,
    out: Optional[npt.NDArray[np.uint8]] = None,
    use_mmap: bool = False,
) -> Union[npt.NDArray[np.uint8], Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]]: ...
async def write_async(
    path: Union[str, Path],
    image: npt.NDArray[np.uint8],
    effort: int = 7,
    distance: float = 1.0,
    lossless: bool = False,
    decoding_speed: int = 0,
    *,
    exif: Optional[bytes] = None,
    xmp: Optional[bytes] = None,
    jumbf: Optional[bytes] = None,
    icc: Optional[bytes] = None,
) -> None: ...

class JXL(_JXL):
    @overload
    def read(
        self,
        path: Union[str, Path],
        *,
        metadata: Literal[False] = False,
        out: Optional[npt.NDArray[np.uint8]] = None,
        use_mmap: bool = False,
    ) -> npt.NDArray[np.uint8]: ...
    @overload
    def read(
        self,
        path: Union[str, Path],
        *,
        metadata: Literal[True],
        out: Optional[npt.NDArray[np.uint8]] = None,
        use_mmap: bool = False,
    ) -> Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]: ...
    @overload
    def read(
        self,
        path: Union[str, Path],
        *,
        metadata: bool,
        out: Optional[npt.NDArray[np.uint8]] = None,
        use_mmap: bool = False,
    ) -> Union[
        npt.NDArray[np.uint8], Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]
    ]: ...
    def write(
        self,
        path: Union[str, Path],
        image: npt.NDArray[np.uint8],
        effort: Optional[int] = None,
        distance: Optional[float] = None,
        lossless: Optional[bool] = None,
        decoding_speed: Optional[int] = None,
        *,
        exif: Optional[bytes] = None,
        xmp: Optional[bytes] = None,
        jumbf: Optional[bytes] = None,
        icc: Optional[bytes] = None,
    ) -> None: ...
    def read_jpeg(
        self,
        path: Union[str, Path],
        *,
        out: Optional[npt.NDArray[np.uint8]] = None,
        use_mmap: bool = False,
    ) -> npt.NDArray[np.uint8]: ...
    def write_jpeg(
        self, path: Union[str, Path], image: npt.NDArray[np.uint8], quality: int = 95
    ) -> None: ...
    def convert_jpeg_to_jxl(
        self,
        jpeg_path: Union[str, Path],
        jxl_path: Union[str, Path],
        effort: Optional[int] = None,
    ) -> None: ...
    def convert_jxl_to_jpeg(
        self, jxl_path: Union[str, Path], jpeg_path: Union[str, Path]
    ) -> None: ...
    def __enter__(self) -> "JXL": ...

class AsyncJXL(_JXL):
    async def __aenter__(self) -> "AsyncJXL": ...
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    async def encode_async(
        self,
        input: npt.NDArray[np.uint8],
        effort: Optional[int] = None,
        distance: Optional[float] = None,
        lossless: Optional[bool] = None,
        decoding_speed: Optional[int] = None,
        *,
        exif: Optional[bytes] = None,
        xmp: Optional[bytes] = None,
        jumbf: Optional[bytes] = None,
        icc: Optional[bytes] = None,
    ) -> bytes: ...
    @overload
    async def decode_async(
        self,
        data: BufferType,
        *,
        metadata: Literal[False] = False,
        out: Optional[npt.NDArray[np.uint8]] = None,
    ) -> npt.NDArray[np.uint8]: ...
    @overload
    async def decode_async(
        self,
        data: BufferType,
        *,
        metadata: Literal[True],
        out: Optional[npt.NDArray[np.uint8]] = None,
    ) -> Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]: ...
    @overload
    async def decode_async(
        self,
        data: BufferType,
        *,
        metadata: bool,
        out: Optional[npt.NDArray[np.uint8]] = None,
    ) -> Union[
        npt.NDArray[np.uint8], Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]
    ]: ...
    @overload
    async def read_async(
        self,
        path: Union[str, Path],
        *,
        metadata: Literal[False] = False,
        out: Optional[npt.NDArray[np.uint8]] = None,
        use_mmap: bool = False,
    ) -> npt.NDArray[np.uint8]: ...
    @overload
    async def read_async(
        self,
        path: Union[str, Path],
        *,
        metadata: Literal[True],
        out: Optional[npt.NDArray[np.uint8]] = None,
        use_mmap: bool = False,
    ) -> Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]: ...
    @overload
    async def read_async(
        self,
        path: Union[str, Path],
        *,
        metadata: bool,
        out: Optional[npt.NDArray[np.uint8]] = None,
        use_mmap: bool = False,
    ) -> Union[
        npt.NDArray[np.uint8], Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]
    ]: ...
    async def write_async(
        self,
        path: Union[str, Path],
        image: npt.NDArray[np.uint8],
        effort: Optional[int] = None,
        distance: Optional[float] = None,
        lossless: Optional[bool] = None,
        decoding_speed: Optional[int] = None,
        *,
        exif: Optional[bytes] = None,
        xmp: Optional[bytes] = None,
        jumbf: Optional[bytes] = None,
        icc: Optional[bytes] = None,
    ) -> None: ...
    async def encode_jpeg_async(
        self, input: npt.NDArray[np.uint8], quality: int = 95
    ) -> bytes: ...
    async def decode_jpeg_async(
        self, data: BufferType, *, out: Optional[npt.NDArray[np.uint8]] = None
    ) -> npt.NDArray[np.uint8]: ...
    async def read_jpeg_async(
        self,
        path: Union[str, Path],
        *,
        out: Optional[npt.NDArray[np.uint8]] = None,
        use_mmap: bool = False,
    ) -> npt.NDArray[np.uint8]: ...
    async def write_jpeg_async(
        self, path: Union[str, Path], image: npt.NDArray[np.uint8], quality: int = 95
    ) -> None: ...
    async def jpeg_to_jxl_async(
        self, data: BufferType, effort: Optional[int] = None
    ) -> bytes: ...
    async def jxl_to_jpeg_async(self, data: BufferType) -> bytes: ...
    async def convert_jpeg_to_jxl_async(
        self,
        jpeg_path: Union[str, Path],
        jxl_path: Union[str, Path],
        effort: Optional[int] = None,
    ) -> None: ...
    async def convert_jxl_to_jpeg_async(
        self, jxl_path: Union[str, Path], jpeg_path: Union[str, Path]
    ) -> None: ...

async def encode_jpeg_async(
    input: npt.NDArray[np.uint8], quality: int = 95
) -> bytes: ...
async def decode_jpeg_async(
    data: BufferType, *, out: Optional[npt.NDArray[np.uint8]] = None
) -> npt.NDArray[np.uint8]: ...
async def jpeg_to_jxl_async(data: BufferType, effort: int = 7) -> bytes: ...
async def jxl_to_jpeg_async(data: BufferType) -> bytes: ...
def read_jpeg(
    path: Union[str, Path],
    *,
    out: Optional[npt.NDArray[np.uint8]] = None,
    use_mmap: bool = False,
) -> npt.NDArray[np.uint8]: ...
def write_jpeg(
    path: Union[str, Path], image: npt.NDArray[np.uint8], quality: int = 95
) -> None: ...
async def read_jpeg_async(
    path: Union[str, Path],
    *,
    out: Optional[npt.NDArray[np.uint8]] = None,
    use_mmap: bool = False,
) -> npt.NDArray[np.uint8]: ...
async def write_jpeg_async(
    path: Union[str, Path], image: npt.NDArray[np.uint8], quality: int = 95
) -> None: ...
def convert_jpeg_to_jxl(
    jpeg_path: Union[str, Path], jxl_path: Union[str, Path], effort: int = 7
) -> None: ...
def convert_jxl_to_jpeg(
    jxl_path: Union[str, Path], jpeg_path: Union[str, Path]
) -> None: ...
async def convert_jpeg_to_jxl_async(
    jpeg_path: Union[str, Path], jxl_path: Union[str, Path], effort: int = 7
) -> None: ...
async def convert_jxl_to_jpeg_async(
    jxl_path: Union[str, Path], jpeg_path: Union[str, Path]
) -> None: ...
