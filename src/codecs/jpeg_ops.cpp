#include "codecs/jpeg_ops.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <turbojpeg.h>

#include "common/deleters.hpp"

namespace nb = nanobind;

namespace pylibjxl {

nb::bytes encode_jpeg(nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu> input, int quality) {
  if (input.ndim() != 3) {
    throw std::invalid_argument("Input must be a 3D array (height, width, channels)");
  }
  const auto height = static_cast<int>(input.shape(0));
  const auto width = static_cast<int>(input.shape(1));
  const auto channels = static_cast<int>(input.shape(2));

  if (channels != 3 && channels != 4) {
    throw std::invalid_argument("Input must have 3 (RGB) or 4 (RGBA) channels");
  }

  quality = std::clamp(quality, 1, 100);

  const auto *input_ptr = static_cast<const uint8_t *>(input.data());

  unsigned char *jpeg_buf = nullptr;
  unsigned long jpeg_size = 0; // NOLINT(google-runtime-int)

  {
    nb::gil_scoped_release release;

    TjPtr compressor(tjInitCompress());
    if (compressor == nullptr) {
      throw std::runtime_error("tjInitCompress failed");
    }

    int pixel_format = (channels == 3) ? TJPF_RGB : TJPF_RGBA;
    int subsamp = TJSAMP_444;

    if (tjCompress2(compressor.get(),
                    static_cast<const unsigned char *>(input_ptr),
                    width,
                    0,
                    height,
                    pixel_format,
                    &jpeg_buf,
                    &jpeg_size,
                    subsamp,
                    quality,
                    TJFLAG_FASTDCT) != 0) {
      throw std::runtime_error(std::string("tjCompress2 failed: ") +
                               tjGetErrorStr2(compressor.get()));
    }
  }

  TjBufPtr guard(jpeg_buf);
  return nb::bytes(reinterpret_cast<const char *>(jpeg_buf), jpeg_size);
}

nb::ndarray<uint8_t, nb::numpy, nb::device::cpu> decode_jpeg(nb::bytes data) {
  char *raw_ptr = nullptr;
  Py_ssize_t raw_size = 0;
  if (PyBytes_AsStringAndSize(data.ptr(), &raw_ptr, &raw_size) != 0) {
    throw nb::python_error();
  }

  const auto *jpeg_data = reinterpret_cast<const unsigned char *>(raw_ptr);
  const auto jpeg_size = static_cast<unsigned long>(raw_size); // NOLINT

  int width = 0;
  int height = 0;
  int subsamp = 0;
  int colorspace = 0;

  std::unique_ptr<uint8_t[]> temp_owner;
  uint8_t *result_ptr_var = nullptr;

  {
    nb::gil_scoped_release release;
    TjPtr decompressor(tjInitDecompress());
    if (decompressor == nullptr) {
      throw std::runtime_error("tjInitDecompress failed");
    }
    if (tjDecompressHeader3(
            decompressor.get(), jpeg_data, jpeg_size, &width, &height, &subsamp, &colorspace) !=
        0) {
      throw std::runtime_error(std::string("tjDecompressHeader3 failed: ") +
                               tjGetErrorStr2(decompressor.get()));
    }

    temp_owner.reset(new uint8_t[height * width * 3]);
    result_ptr_var = temp_owner.get();

    if (tjDecompress2(decompressor.get(),
                      jpeg_data,
                      jpeg_size,
                      static_cast<unsigned char *>(result_ptr_var),
                      width,
                      0,
                      height,
                      TJPF_RGB,
                      TJFLAG_FASTDCT) != 0) {
      throw std::runtime_error(std::string("tjDecompress2 failed: ") +
                               tjGetErrorStr2(decompressor.get()));
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  size_t shape[3] = {static_cast<size_t>(height), static_cast<size_t>(width), 3};
  nb::capsule owner(result_ptr_var, [](void *p) noexcept { delete[] static_cast<uint8_t *>(p); });
  temp_owner.release();

  return nb::ndarray<uint8_t, nb::numpy, nb::device::cpu>(result_ptr_var, 3, shape, owner);
}

} // namespace pylibjxl
