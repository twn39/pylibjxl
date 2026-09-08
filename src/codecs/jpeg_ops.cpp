#include "codecs/jpeg_ops.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <turbojpeg.h>

#include "common/deleters.hpp"
#include "common/utils.hpp"

namespace nb = nanobind;

namespace pylibjxl {

nb::bytes encode_jpeg(nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu> input, int quality) {
  if (input.ndim() != 2 && input.ndim() != 3) {
    throw std::invalid_argument(
        "Input must be a 2D (height, width) or 3D (height, width, channels) array, got ndim=" +
        std::to_string(input.ndim()));
  }
  const auto height = static_cast<int>(input.shape(0));
  const auto width = static_cast<int>(input.shape(1));
  const auto channels = input.ndim() == 2 ? 1 : static_cast<int>(input.shape(2));

  if (channels != 1 && channels != 3 && channels != 4) {
    throw std::invalid_argument("Input must have 1 (Grayscale), 3 (RGB), or 4 (RGBA) channels");
  }

  quality = std::clamp(quality, 1, 100);

  const auto *input_ptr = static_cast<const uint8_t *>(input.data());
  int pixel_format = (channels == 1) ? TJPF_GRAY : (channels == 3 ? TJPF_RGB : TJPF_RGBA);
  int subsamp = (channels == 1) ? TJSAMP_GRAY : TJSAMP_444;

  unsigned char *jpeg_buf = nullptr;
  unsigned long jpeg_size = 0; // NOLINT(google-runtime-int)

  {
    nb::gil_scoped_release release;

    TjPtr compressor(tjInitCompress());
    if (compressor == nullptr) {
      throw std::runtime_error("tjInitCompress failed");
    }

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

nb::object decode_jpeg(nb::handle data,
                       std::optional<nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu>> out) {
  ScopedPyBuffer py_buf(data);
  const auto *jpeg_data = py_buf.data();
  const auto jpeg_size = static_cast<unsigned long>(py_buf.size());

  int width = 0;
  int height = 0;
  int subsamp = 0;
  int colorspace = 0;

  std::unique_ptr<uint8_t[]> temp_owner;
  uint8_t *result_ptr_var = nullptr;
  int out_channels = 3;
  int pixel_format = TJPF_RGB;

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

    if (out.has_value()) {
      if (out->ndim() == 2 && out->shape(0) == static_cast<size_t>(height) &&
          out->shape(1) == static_cast<size_t>(width)) {
        out_channels = 1;
        pixel_format = TJPF_GRAY;
        result_ptr_var = static_cast<uint8_t *>(out->data());
      } else if (out->ndim() == 3 && out->shape(0) == static_cast<size_t>(height) &&
                 out->shape(1) == static_cast<size_t>(width) && out->shape(2) == 1) {
        out_channels = 1;
        pixel_format = TJPF_GRAY;
        result_ptr_var = static_cast<uint8_t *>(out->data());
      } else if (out->ndim() == 3 && out->shape(0) == static_cast<size_t>(height) &&
                 out->shape(1) == static_cast<size_t>(width) && out->shape(2) == 3) {
        out_channels = 3;
        pixel_format = TJPF_RGB;
        result_ptr_var = static_cast<uint8_t *>(out->data());
      } else {
        throw std::invalid_argument("Output buffer shape does not match JPEG dimensions (" +
                                    std::to_string(height) + ", " + std::to_string(width) + ")");
      }
    } else {
      if (subsamp == TJSAMP_GRAY || colorspace == TJCS_GRAY) {
        out_channels = 1;
        pixel_format = TJPF_GRAY;
      } else {
        out_channels = 3;
        pixel_format = TJPF_RGB;
      }
      temp_owner.reset(new uint8_t[height * width * out_channels]);
      result_ptr_var = temp_owner.get();
    }

    if (tjDecompress2(decompressor.get(),
                      jpeg_data,
                      jpeg_size,
                      static_cast<unsigned char *>(result_ptr_var),
                      width,
                      0,
                      height,
                      pixel_format,
                      TJFLAG_FASTDCT) != 0) {
      throw std::runtime_error(std::string("tjDecompress2 failed: ") +
                               tjGetErrorStr2(decompressor.get()));
    }
  }

  if (out.has_value()) {
    return nb::cast(*out);
  }

  nb::capsule owner(result_ptr_var, [](void *p) noexcept { delete[] static_cast<uint8_t *>(p); });
  temp_owner.release();

  if (out_channels == 1) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    size_t shape[2] = {static_cast<size_t>(height), static_cast<size_t>(width)};
    return nb::cast(
        nb::ndarray<uint8_t, nb::numpy, nb::device::cpu>(result_ptr_var, 2, shape, owner));
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    size_t shape[3] = {
        static_cast<size_t>(height), static_cast<size_t>(width), static_cast<size_t>(out_channels)};
    return nb::cast(
        nb::ndarray<uint8_t, nb::numpy, nb::device::cpu>(result_ptr_var, 3, shape, owner));
  }
}

} // namespace pylibjxl
