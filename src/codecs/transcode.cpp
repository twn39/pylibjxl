#include "codecs/transcode.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/resizable_parallel_runner.h>

#include "common/deleters.hpp"
#include "common/utils.hpp"

namespace nb = nanobind;

namespace pylibjxl {

namespace {

std::vector<uint8_t> read_file_bytes(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("No such file or cannot open: " + path);
  }
  auto size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
    throw std::runtime_error("Failed to read file: " + path);
  }
  return buffer;
}

void write_file_bytes(const std::string &path, const uint8_t *data, size_t size) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file for writing: " + path);
  }
  if (!file.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(size))) {
    throw std::runtime_error("Failed to write file: " + path);
  }
}

std::vector<uint8_t>
transcode_jpeg_to_jxl_raw(const uint8_t *jpeg_ptr, size_t jpeg_len, int effort, RunnerPool &pool) {
  effort = std::clamp(effort, 1, 11);

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

  std::vector<uint8_t> compressed(jpeg_len + 4096);
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
  return compressed;
}

std::vector<uint8_t>
transcode_jxl_to_jpeg_raw(const uint8_t *jxl_ptr, size_t jxl_len, RunnerPool &pool) {
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
  std::vector<uint8_t> jpeg_data(k_initial_size);
  size_t jpeg_pos = 0;
  bool reconstruction_seen = false;

  for (int i = 0; i < 1000; ++i) {
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
  return jpeg_data;
}

} // namespace

nb::bytes jpeg_to_jxl_impl(nb::handle jpeg_data, int effort, RunnerPool &pool) {
  ScopedPyBuffer py_buf(jpeg_data);
  const auto *jpeg_ptr = py_buf.data();
  const auto jpeg_len = py_buf.size();

  std::vector<uint8_t> compressed;
  {
    nb::gil_scoped_release release;
    compressed = transcode_jpeg_to_jxl_raw(jpeg_ptr, jpeg_len, effort, pool);
  }
  return nb::bytes(reinterpret_cast<const char *>(compressed.data()), compressed.size());
}

nb::bytes jpeg_to_jxl(nb::handle jpeg_data, int effort) {
  return jpeg_to_jxl_impl(jpeg_data, effort, global_pool());
}

nb::bytes jxl_to_jpeg_impl(nb::handle jxl_data, RunnerPool &pool) {
  ScopedPyBuffer py_buf(jxl_data);
  const auto *jxl_ptr = py_buf.data();
  const auto jxl_len = py_buf.size();

  std::vector<uint8_t> jpeg_data;
  {
    nb::gil_scoped_release release;
    jpeg_data = transcode_jxl_to_jpeg_raw(jxl_ptr, jxl_len, pool);
  }
  return nb::bytes(reinterpret_cast<const char *>(jpeg_data.data()), jpeg_data.size());
}

nb::bytes jxl_to_jpeg(nb::handle jxl_data) {
  return jxl_to_jpeg_impl(jxl_data, global_pool());
}

void jpeg_to_jxl_file_impl(const std::string &in_path,
                           const std::string &out_path,
                           int effort,
                           RunnerPool &pool) {
  nb::gil_scoped_release release;
  std::vector<uint8_t> in_data = read_file_bytes(in_path);
  std::vector<uint8_t> out_data =
      transcode_jpeg_to_jxl_raw(in_data.data(), in_data.size(), effort, pool);
  write_file_bytes(out_path, out_data.data(), out_data.size());
}

void jpeg_to_jxl_file(const std::string &in_path, const std::string &out_path, int effort) {
  jpeg_to_jxl_file_impl(in_path, out_path, effort, global_pool());
}

void jxl_to_jpeg_file_impl(const std::string &in_path,
                           const std::string &out_path,
                           RunnerPool &pool) {
  nb::gil_scoped_release release;
  std::vector<uint8_t> in_data = read_file_bytes(in_path);
  std::vector<uint8_t> out_data = transcode_jxl_to_jpeg_raw(in_data.data(), in_data.size(), pool);
  write_file_bytes(out_path, out_data.data(), out_data.size());
}

void jxl_to_jpeg_file(const std::string &in_path, const std::string &out_path) {
  jxl_to_jpeg_file_impl(in_path, out_path, global_pool());
}

} // namespace pylibjxl
