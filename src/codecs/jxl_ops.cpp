#include "codecs/jxl_ops.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <jxl/cms.h>
#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/resizable_parallel_runner.h>

#include "common/deleters.hpp"
#include "common/utils.hpp"

namespace nb = nanobind;

namespace pylibjxl {

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
nb::bytes encode_impl(nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu> input,
                      int effort,
                      float distance,
                      bool lossless,
                      int decoding_speed,
                      nb::handle exif,
                      nb::handle xmp,
                      nb::handle jumbf,
                      nb::handle icc,
                      RunnerPool &pool) {
  if (input.ndim() != 2 && input.ndim() != 3) {
    throw std::invalid_argument(
        "Input must be a 2D (height, width) or 3D (height, width, channels) array, got ndim=" +
        std::to_string(input.ndim()));
  }

  const auto height = static_cast<size_t>(input.shape(0));
  const auto width = static_cast<size_t>(input.shape(1));
  const auto channels = input.ndim() == 2 ? 1 : static_cast<size_t>(input.shape(2));

  if (channels != 1 && channels != 3 && channels != 4) {
    throw std::invalid_argument(
        "Input must have 1 (Grayscale), 3 (RGB), or 4 (RGBA) channels, got " +
        std::to_string(channels));
  }

  // Extract metadata bytes while GIL is held to avoid data races with Python GC
  std::vector<uint8_t> exif_data = extract_optional_bytes(exif);
  std::vector<uint8_t> xmp_data = extract_optional_bytes(xmp);
  std::vector<uint8_t> jumbf_data = extract_optional_bytes(jumbf);
  std::vector<uint8_t> icc_data = extract_optional_bytes(icc);
  const bool has_metadata =
      !exif_data.empty() || !xmp_data.empty() || !jumbf_data.empty() || !icc_data.empty();

  effort = std::clamp(effort, 1, 11);
  decoding_speed = std::clamp(decoding_speed, 0, 4);
  distance = lossless ? 0.0F : std::clamp(distance, 0.0F, 25.0F);

  const auto *input_ptr = static_cast<const uint8_t *>(input.data());
  const auto input_size = static_cast<size_t>(input.size() * sizeof(uint8_t));

  std::vector<uint8_t> compressed;
  {
    nb::gil_scoped_release release;
    RunnerGuard guard(pool);
    void *runner = guard.get();

    JxlEncoderPtr enc(JxlEncoderCreate(nullptr));
    if (enc == nullptr) {
      throw std::runtime_error("JxlEncoderCreate failed");
    }

    if (runner != nullptr) {
      if (JXL_ENC_SUCCESS !=
          JxlEncoderSetParallelRunner(enc.get(), JxlResizableParallelRunner, runner)) {
        throw std::runtime_error("JxlEncoderSetParallelRunner failed");
      }
    }

    if (has_metadata) {
      if (JXL_ENC_SUCCESS != JxlEncoderUseBoxes(enc.get())) {
        throw std::runtime_error("JxlEncoderUseBoxes failed");
      }
    }

    if (effort > 9) {
      JxlEncoderAllowExpertOptions(enc.get());
    }

    JxlEncoderFrameSettings *frame_settings = JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
    JxlEncoderFrameSettingsSetOption(frame_settings, JXL_ENC_FRAME_SETTING_EFFORT, effort);
    JxlEncoderFrameSettingsSetOption(
        frame_settings, JXL_ENC_FRAME_SETTING_DECODING_SPEED, decoding_speed);

    if (lossless) {
      JxlEncoderSetFrameLossless(frame_settings, JXL_TRUE);
    } else {
      JxlEncoderSetFrameDistance(frame_settings, distance);
    }

    JxlBasicInfo basic_info;
    JxlEncoderInitBasicInfo(&basic_info);
    basic_info.xsize = static_cast<uint32_t>(width);
    basic_info.ysize = static_cast<uint32_t>(height);
    basic_info.bits_per_sample = 8;
    basic_info.uses_original_profile = JXL_TRUE;
    basic_info.num_color_channels = (channels == 1) ? 1 : 3;
    if (channels == 4) {
      basic_info.num_extra_channels = 1;
      basic_info.alpha_bits = 8;
    }

    if (JXL_ENC_SUCCESS != JxlEncoderSetBasicInfo(enc.get(), &basic_info)) {
      throw std::runtime_error("JxlEncoderSetBasicInfo failed");
    }

    if (!icc_data.empty()) {
      if (JXL_ENC_SUCCESS != JxlEncoderSetICCProfile(enc.get(), icc_data.data(), icc_data.size())) {
        throw std::runtime_error("JxlEncoderSetICCProfile failed");
      }
    } else {
      JxlColorEncoding color_encoding = {};
      JxlColorEncodingSetToSRGB(&color_encoding, channels == 1 ? JXL_TRUE : JXL_FALSE);
      if (JXL_ENC_SUCCESS != JxlEncoderSetColorEncoding(enc.get(), &color_encoding)) {
        throw std::runtime_error("JxlEncoderSetColorEncoding failed");
      }
    }

    JxlPixelFormat pixel_format = {
        static_cast<uint32_t>(channels), JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0};

    if (JXL_ENC_SUCCESS !=
        JxlEncoderAddImageFrame(frame_settings, &pixel_format, input_ptr, input_size)) {
      throw std::runtime_error("JxlEncoderAddImageFrame failed");
    }

    if (has_metadata) {
      JxlEncoderCloseFrames(enc.get());

      if (!exif_data.empty()) {
        // EXIF box requires 4-byte TIFF header offset prefix (usually 0) to comply with JXL spec
        std::vector<uint8_t> exif_box(4 + exif_data.size(), 0);
        std::memcpy(exif_box.data() + 4, exif_data.data(), exif_data.size());
        if (JXL_ENC_SUCCESS !=
            JxlEncoderAddBox(enc.get(), "Exif", exif_box.data(), exif_box.size(), JXL_TRUE)) {
          throw std::runtime_error("JxlEncoderAddBox(Exif) failed");
        }
      }

      if (!xmp_data.empty()) {
        if (JXL_ENC_SUCCESS !=
            JxlEncoderAddBox(enc.get(), "xml ", xmp_data.data(), xmp_data.size(), JXL_TRUE)) {
          throw std::runtime_error("JxlEncoderAddBox(XMP) failed");
        }
      }

      if (!jumbf_data.empty()) {
        if (JXL_ENC_SUCCESS !=
            JxlEncoderAddBox(enc.get(), "jumb", jumbf_data.data(), jumbf_data.size(), JXL_TRUE)) {
          throw std::runtime_error("JxlEncoderAddBox(JUMBF) failed");
        }
      }

      JxlEncoderCloseBoxes(enc.get());
    } else {
      JxlEncoderCloseInput(enc.get());
    }

    const size_t estimated = std::max<size_t>(width * height * channels / 2, 4096);
    compressed.resize(estimated);
    uint8_t *next_out = compressed.data();
    size_t avail_out = compressed.size();

    JxlEncoderStatus status = JXL_ENC_NEED_MORE_OUTPUT;
    while (status == JXL_ENC_NEED_MORE_OUTPUT) {
      status = JxlEncoderProcessOutput(enc.get(), &next_out, &avail_out);
      if (status == JXL_ENC_NEED_MORE_OUTPUT) {
        const size_t offset = static_cast<size_t>(next_out - compressed.data());
        compressed.resize(compressed.size() * 2);
        next_out = compressed.data() + offset;
        avail_out = compressed.size() - offset;
      }
    }
    if (status != JXL_ENC_SUCCESS) {
      throw std::runtime_error("JxlEncoderProcessOutput failed");
    }
    compressed.resize(static_cast<size_t>(next_out - compressed.data()));
  }

  return nb::bytes(reinterpret_cast<const char *>(compressed.data()), compressed.size());
}

nb::bytes encode(nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu> input,
                 int effort,
                 float distance,
                 bool lossless,
                 int decoding_speed,
                 nb::handle exif,
                 nb::handle xmp,
                 nb::handle jumbf,
                 nb::handle icc) {
  return encode_impl(
      input, effort, distance, lossless, decoding_speed, exif, xmp, jumbf, icc, global_pool());
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
nb::object decode_impl(nb::handle data,
                       bool metadata,
                       std::optional<nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu>> out,
                       RunnerPool &pool) {
  ScopedPyBuffer py_buf(data);
  const auto *jxl_data = py_buf.data();
  const auto jxl_size = py_buf.size();

  JxlBasicInfo info{};
  size_t channels = 0;
  std::unique_ptr<uint8_t[]> temp_owner;
  uint8_t *result_ptr_var = nullptr;
  std::map<std::string, std::vector<uint8_t>> boxes;
  std::vector<uint8_t> icc_data;

  {
    nb::gil_scoped_release release;
    RunnerGuard guard(pool);
    void *runner = guard.get();

    JxlDecoderPtr dec(JxlDecoderCreate(nullptr));
    if (dec == nullptr) {
      throw std::runtime_error("JxlDecoderCreate failed");
    }

    if (JXL_DEC_SUCCESS != JxlDecoderSetCms(dec.get(), *JxlGetDefaultCms())) {
      throw std::runtime_error("JxlDecoderSetCms failed");
    }

    if (runner != nullptr) {
      if (JXL_DEC_SUCCESS !=
          JxlDecoderSetParallelRunner(dec.get(), JxlResizableParallelRunner, runner)) {
        throw std::runtime_error("JxlDecoderSetParallelRunner failed");
      }
    }

    int events = JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE |
                 (metadata ? (JXL_DEC_BOX | JXL_DEC_COLOR_ENCODING) : 0);
    if (metadata) {
      JxlDecoderSetDecompressBoxes(dec.get(), JXL_TRUE);
    }
    if (JXL_DEC_SUCCESS != JxlDecoderSubscribeEvents(dec.get(), events)) {
      throw std::runtime_error("JxlDecoderSubscribeEvents failed");
    }

    JxlDecoderSetInput(dec.get(), jxl_data, jxl_size);
    JxlDecoderCloseInput(dec.get());

    JxlPixelFormat format = {};
    std::string current_box_type;
    std::vector<uint8_t> box_buffer;
    constexpr size_t k_box_chunk_size = 65536;

    for (;;) {
      JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());

      if (status == JXL_DEC_ERROR) {
        throw std::runtime_error("Decoder error during pixel decode");
      }
      if (status == JXL_DEC_NEED_MORE_INPUT) {
        throw std::runtime_error("Truncated JXL data: need more input for pixels");
      }
      if (status == JXL_DEC_COLOR_ENCODING) {
        JxlColorEncoding color_encoding = {};
        JxlDecoderStatus enc_status = JxlDecoderGetColorAsEncodedProfile(
            dec.get(), JXL_COLOR_PROFILE_TARGET_ORIGINAL, &color_encoding);
        // Only extract ICC profile if the image does not use a structured profile
        if (enc_status != JXL_DEC_SUCCESS) {
          size_t icc_size = 0;
          if (JXL_DEC_SUCCESS == JxlDecoderGetICCProfileSize(
                                     dec.get(), JXL_COLOR_PROFILE_TARGET_ORIGINAL, &icc_size) &&
              icc_size > 0) {
            icc_data.resize(icc_size);
            if (JXL_DEC_SUCCESS != JxlDecoderGetColorAsICCProfile(dec.get(),
                                                                  JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                                                                  icc_data.data(),
                                                                  icc_data.size())) {
              icc_data.clear();
            }
          }
        }
        continue;
      }
      if (status == JXL_DEC_BASIC_INFO) {
        if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(dec.get(), &info)) {
          throw std::runtime_error("JxlDecoderGetBasicInfo failed");
        }
        channels = info.num_color_channels + (info.alpha_bits > 0 ? 1 : 0);
        format = {static_cast<uint32_t>(channels), JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0};
        continue;
      }
      if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
        const size_t result_bytes = static_cast<size_t>(info.ysize * info.xsize * channels);
        if (out.has_value()) {
          bool shape_match = false;
          if (out->ndim() == 3 && out->shape(0) == info.ysize && out->shape(1) == info.xsize &&
              out->shape(2) == channels) {
            shape_match = true;
          } else if (out->ndim() == 2 && channels == 1 && out->shape(0) == info.ysize &&
                     out->shape(1) == info.xsize) {
            shape_match = true;
          }
          if (!shape_match) {
            throw std::invalid_argument(
                "Output buffer shape does not match image shape (" + std::to_string(info.ysize) +
                ", " + std::to_string(info.xsize) + ", " + std::to_string(channels) + ")");
          }
          result_ptr_var = static_cast<uint8_t *>(out->data());
        } else {
          temp_owner.reset(new uint8_t[result_bytes]);
          result_ptr_var = temp_owner.get();
        }
        if (JXL_DEC_SUCCESS !=
            JxlDecoderSetImageOutBuffer(dec.get(), &format, result_ptr_var, result_bytes)) {
          throw std::runtime_error("JxlDecoderSetImageOutBuffer failed");
        }
        continue;
      }
      if (status == JXL_DEC_BOX) {
        if (!current_box_type.empty()) {
          size_t remaining = JxlDecoderReleaseBoxBuffer(dec.get());
          box_buffer.resize(box_buffer.size() - remaining);
          boxes[current_box_type] = std::move(box_buffer);
          current_box_type.clear();
        }

        JxlBoxType box_type{};
        if (JXL_DEC_SUCCESS != JxlDecoderGetBoxType(dec.get(), box_type, JXL_TRUE)) {
          continue;
        }
        std::string type_str(box_type, 4);

        if (type_str == "Exif" || type_str == "xml " || type_str == "jumb") {
          current_box_type = type_str;
          box_buffer.resize(k_box_chunk_size);
          JxlDecoderSetBoxBuffer(dec.get(), box_buffer.data(), box_buffer.size());
        }
        continue;
      }
      if (status == JXL_DEC_BOX_NEED_MORE_OUTPUT) {
        size_t remaining = JxlDecoderReleaseBoxBuffer(dec.get());
        size_t bytes_read = box_buffer.size() - remaining;
        box_buffer.resize(box_buffer.size() + k_box_chunk_size);
        JxlDecoderSetBoxBuffer(
            dec.get(), box_buffer.data() + bytes_read, box_buffer.size() - bytes_read);
        continue;
      }
      if (status == JXL_DEC_FULL_IMAGE) {
        if (!metadata) {
          break;
        }
        continue;
      }
      if (status == JXL_DEC_SUCCESS) {
        if (!current_box_type.empty()) {
          size_t remaining = JxlDecoderReleaseBoxBuffer(dec.get());
          box_buffer.resize(box_buffer.size() - remaining);
          boxes[current_box_type] = std::move(box_buffer);
        }
        break;
      }
    }
  }

  nb::object py_result;
  if (out.has_value()) {
    py_result = nb::cast(*out);
  } else {
    nb::capsule owner(result_ptr_var, [](void *p) noexcept { delete[] static_cast<uint8_t *>(p); });
    temp_owner.release();
    if (channels == 1) {
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
      size_t shape[2] = {static_cast<size_t>(info.ysize), static_cast<size_t>(info.xsize)};
      nb::ndarray<uint8_t, nb::numpy, nb::device::cpu> result(result_ptr_var, 2, shape, owner);
      py_result = nb::cast(result);
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
      size_t shape[3] = {
          static_cast<size_t>(info.ysize), static_cast<size_t>(info.xsize), channels};
      nb::ndarray<uint8_t, nb::numpy, nb::device::cpu> result(result_ptr_var, 3, shape, owner);
      py_result = nb::cast(result);
    }
  }

  if (!metadata) {
    return py_result;
  }

  nb::dict meta;
  if (!icc_data.empty()) {
    meta["icc"] = nb::bytes(reinterpret_cast<const char *>(icc_data.data()), icc_data.size());
    meta["icc_profile"] =
        nb::bytes(reinterpret_cast<const char *>(icc_data.data()), icc_data.size());
  }
  for (auto &[key, value] : boxes) {
    if (key == "Exif" && value.size() > 4) {
      meta["exif"] = nb::bytes(reinterpret_cast<const char *>(value.data() + 4), value.size() - 4);
    } else if (key == "xml ") {
      meta["xmp"] = nb::bytes(reinterpret_cast<const char *>(value.data()), value.size());
    } else if (key == "jumb") {
      meta["jumbf"] = nb::bytes(reinterpret_cast<const char *>(value.data()), value.size());
    }
  }
  return nb::make_tuple(py_result, meta);
}

nb::object decode(nb::handle data,
                  bool metadata,
                  std::optional<nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu>> out) {
  return decode_impl(data, metadata, out, global_pool());
}

} // namespace pylibjxl
