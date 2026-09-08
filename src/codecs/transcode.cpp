#include "codecs/transcode.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/resizable_parallel_runner.h>

#include "common/deleters.hpp"

namespace nb = nanobind;

namespace pylibjxl {

nb::bytes jpeg_to_jxl_impl(nb::bytes jpeg_data, int effort, RunnerPool &pool) {
  char *raw_ptr = nullptr;
  Py_ssize_t raw_size = 0;
  if (PyBytes_AsStringAndSize(jpeg_data.ptr(), &raw_ptr, &raw_size) != 0) {
    throw nb::python_error();
  }
  const auto *jpeg_ptr = reinterpret_cast<const uint8_t *>(raw_ptr);
  const auto jpeg_len = static_cast<size_t>(raw_size);

  effort = std::clamp(effort, 1, 11);

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

    if (JXL_ENC_SUCCESS != JxlEncoderUseContainer(enc.get(), JXL_TRUE)) {
      throw std::runtime_error("JxlEncoderUseContainer failed");
    }

    if (JXL_ENC_SUCCESS != JxlEncoderStoreJPEGMetadata(enc.get(), JXL_TRUE)) {
      throw std::runtime_error("JxlEncoderStoreJPEGMetadata failed");
    }

    JxlEncoderFrameSettings *settings = JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
    if (JXL_ENC_SUCCESS !=
        JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, effort)) {
      throw std::runtime_error("JxlEncoderFrameSettingsSetOption(EFFORT) failed");
    }

    if (JXL_ENC_SUCCESS != JxlEncoderAddJPEGFrame(settings, jpeg_ptr, jpeg_len)) {
      throw std::runtime_error("JxlEncoderAddJPEGFrame failed (input may not be a valid JPEG)");
    }

    JxlEncoderCloseInput(enc.get());

    compressed.resize(jpeg_len + 4096);
    uint8_t *next_out = compressed.data();
    size_t avail_out = compressed.size();
    JxlEncoderStatus status = JXL_ENC_NEED_MORE_OUTPUT;

    while (status == JXL_ENC_NEED_MORE_OUTPUT) {
      status = JxlEncoderProcessOutput(enc.get(), &next_out, &avail_out);
      if (status == JXL_ENC_NEED_MORE_OUTPUT) {
        size_t offset = next_out - compressed.data();
        compressed.resize(compressed.size() * 2);
        next_out = compressed.data() + offset;
        avail_out = compressed.size() - offset;
      }
    }
    if (status != JXL_ENC_SUCCESS) {
      throw std::runtime_error("JxlEncoderProcessOutput failed");
    }
    compressed.resize(next_out - compressed.data());
  }
  return nb::bytes(reinterpret_cast<const char *>(compressed.data()), compressed.size());
}

nb::bytes jpeg_to_jxl(nb::bytes jpeg_data, int effort) {
  return jpeg_to_jxl_impl(jpeg_data, effort, global_pool());
}

nb::bytes jxl_to_jpeg_impl(nb::bytes jxl_data, RunnerPool &pool) {
  char *raw_ptr = nullptr;
  Py_ssize_t raw_size = 0;
  if (PyBytes_AsStringAndSize(jxl_data.ptr(), &raw_ptr, &raw_size) != 0) {
    throw nb::python_error();
  }
  const auto *jxl_ptr = reinterpret_cast<const uint8_t *>(raw_ptr);
  const auto jxl_len = static_cast<size_t>(raw_size);

  std::vector<uint8_t> jpeg_data;
  {
    nb::gil_scoped_release release;
    RunnerGuard guard(pool);
    void *runner = guard.get();

    JxlDecoderPtr dec(JxlDecoderCreate(nullptr));
    if (dec == nullptr) {
      throw std::runtime_error("JxlDecoderCreate failed");
    }

    if (runner != nullptr) {
      if (JXL_DEC_SUCCESS !=
          JxlDecoderSetParallelRunner(dec.get(), JxlResizableParallelRunner, runner)) {
        throw std::runtime_error("JxlDecoderSetParallelRunner failed");
      }
    }

    if (JXL_DEC_SUCCESS !=
        JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_JPEG_RECONSTRUCTION | JXL_DEC_FULL_IMAGE)) {
      throw std::runtime_error("JxlDecoderSubscribeEvents failed");
    }

    JxlDecoderSetInput(dec.get(), jxl_ptr, jxl_len);
    JxlDecoderCloseInput(dec.get());

    constexpr size_t k_initial_size = 4096;
    jpeg_data.resize(k_initial_size);
    size_t jpeg_pos = 0;

    bool reconstruction_seen = false;

    for (int i = 0; i < 1000; ++i) { // Safety limit to prevent infinite loop
      JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());

      if (status == JXL_DEC_ERROR) {
        throw std::runtime_error("JxlDecoderProcessInput failed with JXL_DEC_ERROR");
      }
      if (status == JXL_DEC_SUCCESS) {
        if (reconstruction_seen) {
          size_t remaining = JxlDecoderReleaseJPEGBuffer(dec.get());
          jpeg_pos = jpeg_data.size() - remaining;
        }
        break;
      }
      if (status == JXL_DEC_JPEG_RECONSTRUCTION) {
        reconstruction_seen = true;
        if (JXL_DEC_SUCCESS !=
            JxlDecoderSetJPEGBuffer(dec.get(), jpeg_data.data(), jpeg_data.size())) {
          throw std::runtime_error("JxlDecoderSetJPEGBuffer failed");
        }
        continue;
      }
      if (status == JXL_DEC_JPEG_NEED_MORE_OUTPUT) {
        size_t remaining = JxlDecoderReleaseJPEGBuffer(dec.get());
        jpeg_pos = jpeg_data.size() - remaining;
        jpeg_data.resize(jpeg_data.size() * 2);
        if (JXL_DEC_SUCCESS != JxlDecoderSetJPEGBuffer(dec.get(),
                                                       jpeg_data.data() + jpeg_pos,
                                                       jpeg_data.size() - jpeg_pos)) {
          throw std::runtime_error("JxlDecoderSetJPEGBuffer failed after resize");
        }
        continue;
      }
      if (status == JXL_DEC_NEED_MORE_INPUT) {
        break;
      }
      if (status == JXL_DEC_FULL_IMAGE) {
        continue;
      }

      break;
    }

    if (!reconstruction_seen) {
      throw std::runtime_error("JXL data does not contain a reconstructible JPEG codestream");
    }

    jpeg_data.resize(jpeg_pos);
  }
  return nb::bytes(reinterpret_cast<const char *>(jpeg_data.data()), jpeg_data.size());
}

nb::bytes jxl_to_jpeg(nb::bytes jxl_data) {
  return jxl_to_jpeg_impl(jxl_data, global_pool());
}

} // namespace pylibjxl
