from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union, overload

import numpy as np
import numpy.typing as npt

BufferType = Union[bytes, bytearray, memoryview, npt.NDArray[np.uint8], Any]

# --- Native extension functions ---

class CodecTimeoutError(TimeoutError): ...

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
    timeout: Optional[float] = None,
) -> bytes: ...
@overload
def decode(
    data: BufferType,
    metadata: Literal[False] = False,
    out: Optional[npt.NDArray[np.uint8]] = None,
    timeout: Optional[float] = None,
) -> npt.NDArray[np.uint8]: ...
@overload
def decode(
    data: BufferType,
    metadata: Literal[True],
    out: Optional[npt.NDArray[np.uint8]] = None,
    timeout: Optional[float] = None,
) -> Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]: ...
@overload
def decode(
    data: BufferType,
    metadata: bool,
    out: Optional[npt.NDArray[np.uint8]] = None,
    timeout: Optional[float] = None,
) -> Union[npt.NDArray[np.uint8], Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]]: ...
def encode_jpeg(input: npt.NDArray[np.uint8], quality: int = 95) -> bytes: ...
def decode_jpeg(
    data: BufferType, out: Optional[npt.NDArray[np.uint8]] = None
) -> npt.NDArray[np.uint8]: ...
def probe(data: BufferType) -> Dict[str, Any]: ...
def probe_file(path: Union[str, Path], *, use_mmap: bool = False) -> Dict[str, Any]: ...
def jpeg_to_jxl(
    data: BufferType, effort: int = 7, timeout: Optional[float] = None
) -> bytes: ...
def jxl_to_jpeg(data: BufferType, timeout: Optional[float] = None) -> bytes: ...
def jpeg_to_jxl_file(
    in_path: Union[str, Path],
    out_path: Union[str, Path],
    effort: int = 7,
    timeout: Optional[float] = None,
) -> None: ...
def jxl_to_jpeg_file(
    in_path: Union[str, Path],
    out_path: Union[str, Path],
    timeout: Optional[float] = None,
) -> None: ...

class _JXL:
    def __init__(
        self,
        effort: int = 7,
        distance: float = 1.0,
        lossless: bool = False,
        decoding_speed: int = 0,
        threads: int = 0,
        pool_size: int = 0,
        timeout: float = 0.0,
        idle_timeout: float = 60.0,
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
        timeout: Optional[float] = None,
    ) -> bytes: ...
    @overload
    def decode(
        self,
        data: BufferType,
        metadata: Literal[False] = False,
        out: Optional[npt.NDArray[np.uint8]] = None,
        timeout: Optional[float] = None,
    ) -> npt.NDArray[np.uint8]: ...
    @overload
    def decode(
        self,
        data: BufferType,
        metadata: Literal[True],
        out: Optional[npt.NDArray[np.uint8]] = None,
        timeout: Optional[float] = None,
    ) -> Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]: ...
    @overload
    def decode(
        self,
        data: BufferType,
        metadata: bool,
        out: Optional[npt.NDArray[np.uint8]] = None,
        timeout: Optional[float] = None,
    ) -> Union[
        npt.NDArray[np.uint8], Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]
    ]: ...
    def encode_jpeg(self, input: npt.NDArray[np.uint8], quality: int = 95) -> bytes: ...
    def decode_jpeg(
        self, data: BufferType, out: Optional[npt.NDArray[np.uint8]] = None
    ) -> npt.NDArray[np.uint8]: ...
    def probe(self, data: BufferType) -> Dict[str, Any]: ...
    def probe_file(
        self, path: Union[str, Path], *, use_mmap: bool = False
    ) -> Dict[str, Any]: ...
    def jpeg_to_jxl(
        self,
        data: BufferType,
        effort: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> bytes: ...
    def jxl_to_jpeg(
        self, data: BufferType, timeout: Optional[float] = None
    ) -> bytes: ...
    def jpeg_to_jxl_file(
        self,
        in_path: Union[str, Path],
        out_path: Union[str, Path],
        effort: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None: ...
    def jxl_to_jpeg_file(
        self,
        in_path: Union[str, Path],
        out_path: Union[str, Path],
        timeout: Optional[float] = None,
    ) -> None: ...
    def close(self) -> None: ...
    @property
    def closed(self) -> bool: ...
    @property
    def pool_size(self) -> int: ...
    @property
    def total_runners(self) -> int: ...
    @property
    def available_runners(self) -> int: ...
    @property
    def in_use_runners(self) -> int: ...
    @property
    def threads_per_runner(self) -> int: ...
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
    timeout: Optional[float] = None,
) -> bytes: ...
@overload
async def decode_async(
    data: BufferType,
    *,
    metadata: Literal[False] = False,
    out: Optional[npt.NDArray[np.uint8]] = None,
    timeout: Optional[float] = None,
) -> npt.NDArray[np.uint8]: ...
@overload
async def decode_async(
    data: BufferType,
    *,
    metadata: Literal[True],
    out: Optional[npt.NDArray[np.uint8]] = None,
    timeout: Optional[float] = None,
) -> Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]: ...
@overload
async def decode_async(
    data: BufferType,
    *,
    metadata: bool,
    out: Optional[npt.NDArray[np.uint8]] = None,
    timeout: Optional[float] = None,
) -> Union[npt.NDArray[np.uint8], Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]]: ...
@overload
def read(
    path: Union[str, Path],
    *,
    metadata: Literal[False] = False,
    out: Optional[npt.NDArray[np.uint8]] = None,
    use_mmap: bool = False,
    timeout: Optional[float] = None,
) -> npt.NDArray[np.uint8]: ...
@overload
def read(
    path: Union[str, Path],
    *,
    metadata: Literal[True],
    out: Optional[npt.NDArray[np.uint8]] = None,
    use_mmap: bool = False,
    timeout: Optional[float] = None,
) -> Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]: ...
@overload
def read(
    path: Union[str, Path],
    *,
    metadata: bool,
    out: Optional[npt.NDArray[np.uint8]] = None,
    use_mmap: bool = False,
    timeout: Optional[float] = None,
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
    timeout: Optional[float] = None,
) -> None: ...
@overload
async def read_async(
    path: Union[str, Path],
    *,
    metadata: Literal[False] = False,
    out: Optional[npt.NDArray[np.uint8]] = None,
    use_mmap: bool = False,
    timeout: Optional[float] = None,
) -> npt.NDArray[np.uint8]: ...
@overload
async def read_async(
    path: Union[str, Path],
    *,
    metadata: Literal[True],
    out: Optional[npt.NDArray[np.uint8]] = None,
    use_mmap: bool = False,
    timeout: Optional[float] = None,
) -> Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]: ...
@overload
async def read_async(
    path: Union[str, Path],
    *,
    metadata: bool,
    out: Optional[npt.NDArray[np.uint8]] = None,
    use_mmap: bool = False,
    timeout: Optional[float] = None,
) -> Union[npt.NDArray[np.uint8], Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]]: ...
async def probe_async(data: BufferType) -> Dict[str, Any]: ...
async def probe_file_async(
    path: Union[str, Path], *, use_mmap: bool = False
) -> Dict[str, Any]: ...
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
    timeout: Optional[float] = None,
) -> None: ...

class JXL(_JXL):
    def __init__(
        self,
        effort: int = 7,
        distance: float = 1.0,
        lossless: bool = False,
        decoding_speed: int = 0,
        threads: int = 0,
        pool_size: int = 0,
        timeout: float = 0.0,
        idle_timeout: float = 60.0,
    ) -> None: ...
    @overload
    def read(
        self,
        path: Union[str, Path],
        *,
        metadata: Literal[False] = False,
        out: Optional[npt.NDArray[np.uint8]] = None,
        use_mmap: bool = False,
        timeout: Optional[float] = None,
    ) -> npt.NDArray[np.uint8]: ...
    @overload
    def read(
        self,
        path: Union[str, Path],
        *,
        metadata: Literal[True],
        out: Optional[npt.NDArray[np.uint8]] = None,
        use_mmap: bool = False,
        timeout: Optional[float] = None,
    ) -> Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]: ...
    @overload
    def read(
        self,
        path: Union[str, Path],
        *,
        metadata: bool,
        out: Optional[npt.NDArray[np.uint8]] = None,
        use_mmap: bool = False,
        timeout: Optional[float] = None,
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
        timeout: Optional[float] = None,
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
        timeout: Optional[float] = None,
    ) -> None: ...
    def convert_jxl_to_jpeg(
        self,
        jxl_path: Union[str, Path],
        jpeg_path: Union[str, Path],
        timeout: Optional[float] = None,
    ) -> None: ...
    def __enter__(self) -> "JXL": ...

class AsyncJXL(JXL):
    def __init__(
        self,
        effort: int = 7,
        distance: float = 1.0,
        lossless: bool = False,
        decoding_speed: int = 0,
        threads: int = 0,
        pool_size: int = 0,
        timeout: float = 0.0,
        idle_timeout: float = 60.0,
    ) -> None: ...
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
        timeout: Optional[float] = None,
    ) -> bytes: ...
    @overload
    async def decode_async(
        self,
        data: BufferType,
        *,
        metadata: Literal[False] = False,
        out: Optional[npt.NDArray[np.uint8]] = None,
        timeout: Optional[float] = None,
    ) -> npt.NDArray[np.uint8]: ...
    @overload
    async def decode_async(
        self,
        data: BufferType,
        *,
        metadata: Literal[True],
        out: Optional[npt.NDArray[np.uint8]] = None,
        timeout: Optional[float] = None,
    ) -> Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]: ...
    @overload
    async def decode_async(
        self,
        data: BufferType,
        *,
        metadata: bool,
        out: Optional[npt.NDArray[np.uint8]] = None,
        timeout: Optional[float] = None,
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
        timeout: Optional[float] = None,
    ) -> npt.NDArray[np.uint8]: ...
    @overload
    async def read_async(
        self,
        path: Union[str, Path],
        *,
        metadata: Literal[True],
        out: Optional[npt.NDArray[np.uint8]] = None,
        use_mmap: bool = False,
        timeout: Optional[float] = None,
    ) -> Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]: ...
    @overload
    async def read_async(
        self,
        path: Union[str, Path],
        *,
        metadata: bool,
        out: Optional[npt.NDArray[np.uint8]] = None,
        use_mmap: bool = False,
        timeout: Optional[float] = None,
    ) -> Union[
        npt.NDArray[np.uint8], Tuple[npt.NDArray[np.uint8], Dict[str, bytes]]
    ]: ...
    async def probe_async(self, data: BufferType) -> Dict[str, Any]: ...
    async def probe_file_async(
        self, path: Union[str, Path], *, use_mmap: bool = False
    ) -> Dict[str, Any]: ...
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
        timeout: Optional[float] = None,
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
        self,
        data: BufferType,
        effort: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> bytes: ...
    async def jxl_to_jpeg_async(
        self, data: BufferType, timeout: Optional[float] = None
    ) -> bytes: ...
    async def convert_jpeg_to_jxl_async(
        self,
        jpeg_path: Union[str, Path],
        jxl_path: Union[str, Path],
        effort: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None: ...
    async def convert_jxl_to_jpeg_async(
        self,
        jxl_path: Union[str, Path],
        jpeg_path: Union[str, Path],
        timeout: Optional[float] = None,
    ) -> None: ...

async def encode_jpeg_async(
    input: npt.NDArray[np.uint8], quality: int = 95
) -> bytes: ...
async def decode_jpeg_async(
    data: BufferType, *, out: Optional[npt.NDArray[np.uint8]] = None
) -> npt.NDArray[np.uint8]: ...
async def jpeg_to_jxl_async(
    data: BufferType, effort: int = 7, timeout: Optional[float] = None
) -> bytes: ...
async def jxl_to_jpeg_async(
    data: BufferType, timeout: Optional[float] = None
) -> bytes: ...
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
    jpeg_path: Union[str, Path],
    jxl_path: Union[str, Path],
    effort: int = 7,
    timeout: Optional[float] = None,
) -> None: ...
def convert_jxl_to_jpeg(
    jxl_path: Union[str, Path],
    jpeg_path: Union[str, Path],
    timeout: Optional[float] = None,
) -> None: ...
async def convert_jpeg_to_jxl_async(
    jpeg_path: Union[str, Path],
    jxl_path: Union[str, Path],
    effort: int = 7,
    timeout: Optional[float] = None,
) -> None: ...
async def convert_jxl_to_jpeg_async(
    jxl_path: Union[str, Path],
    jpeg_path: Union[str, Path],
    timeout: Optional[float] = None,
) -> None: ...
