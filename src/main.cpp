#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <turbojpeg.h>
#include <vector>

#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/resizable_parallel_runner.h>
#include <jxl/version.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

// ─── RAII Wrappers
// ─────────────────────────────────────────────────────────────

struct JxlEncoderDeleter {
  void operator()(JxlEncoder *p) const { JxlEncoderDestroy(p); }
};
using JxlEncoderPtr = std::unique_ptr<JxlEncoder, JxlEncoderDeleter>;

struct JxlDecoderDeleter {
  void operator()(JxlDecoder *p) const { JxlDecoderDestroy(p); }
};
using JxlDecoderPtr = std::unique_ptr<JxlDecoder, JxlDecoderDeleter>;

struct JxlRunnerDeleter {
  void operator()(void *p) const { JxlResizableParallelRunnerDestroy(p); }
};
using JxlRunnerPtr = std::unique_ptr<void, JxlRunnerDeleter>;

struct TjDeleter {
  void operator()(void *p) const { tjDestroy(p); }
};
using TjPtr = std::unique_ptr<void, TjDeleter>;

struct TjFree {
  void operator()(unsigned char *p) const { tjFree(p); }
};
using TjBufPtr = std::unique_ptr<unsigned char, TjFree>;

// ─── Encode
// ────────────────────────────────────────────────────────────────────

/// Helper: extract raw bytes from an optional py::object (bytes or None).
static std::vector<uint8_t> extract_optional_bytes(const py::object &obj) {
  if (obj.is_none()) {
    return {};
  }
  auto b = obj.cast<py::bytes>();
  const auto *ptr = PyBytes_AsString(b.ptr());
  const auto size = static_cast<size_t>(PyBytes_Size(b.ptr()));
  return {reinterpret_cast<const uint8_t *>(ptr), reinterpret_cast<const uint8_t *>(ptr) + size};
}

// NOLINTNEXTLINE(misc-use-internal-linkage,cppcoreguidelines-avoid-non-const-global-variables)
py::bytes encode(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> input,
                 int effort = 7,
                 float distance = 1.0F,
                 bool lossless = false,
                 py::object exif = py::none(),
                 py::object xmp = py::none(),
                 py::object jumbf = py::none()) {
  // ── Input validation (GIL held) ──
  auto buf = input.request();
  if (buf.ndim != 3) {
    throw std::invalid_argument("Input must be a 3D array (height, width, channels), got ndim=" +
                                std::to_string(buf.ndim));
  }

  const auto height = static_cast<size_t>(buf.shape[0]);
  const auto width = static_cast<size_t>(buf.shape[1]);
  const auto channels = static_cast<size_t>(buf.shape[2]);

  if (channels != 3 && channels != 4) {
    throw std::invalid_argument("Input must have 3 (RGB) or 4 (RGBA) channels, got " +
                                std::to_string(channels));
  }

  // Extract metadata bytes while GIL is held
  std::vector<uint8_t> exif_data = extract_optional_bytes(exif);
  std::vector<uint8_t> xmp_data = extract_optional_bytes(xmp);
  std::vector<uint8_t> jumbf_data = extract_optional_bytes(jumbf);
  const bool has_metadata = !exif_data.empty() || !xmp_data.empty() || !jumbf_data.empty();

  // Clamp parameters to libjxl valid ranges
  effort = std::clamp(effort, 1, 10);
  distance = lossless ? 0.0F : std::clamp(distance, 0.0F, 25.0F);

  const auto *input_ptr = static_cast<const uint8_t *>(buf.ptr);
  const auto input_size = static_cast<size_t>(buf.size * buf.itemsize);

  // ── Encoding (GIL released) ──
  std::vector<uint8_t> compressed;
  {
    py::gil_scoped_release release;

    JxlRunnerPtr runner(JxlResizableParallelRunnerCreate(nullptr));
    if (!runner) {
      throw std::runtime_error("JxlResizableParallelRunnerCreate failed");
    }
    JxlResizableParallelRunnerSetThreads(runner.get(),
                                         JxlResizableParallelRunnerSuggestThreads(width, height));

    JxlEncoderPtr enc(JxlEncoderCreate(nullptr));
    if (!enc) {
      throw std::runtime_error("JxlEncoderCreate failed");
    }

    if (JXL_ENC_SUCCESS !=
        JxlEncoderSetParallelRunner(enc.get(), JxlResizableParallelRunner, runner.get())) {
      throw std::runtime_error("JxlEncoderSetParallelRunner failed");
    }

    // Enable box-based container if metadata is present
    if (has_metadata) {
      if (JXL_ENC_SUCCESS != JxlEncoderUseBoxes(enc.get())) {
        throw std::runtime_error("JxlEncoderUseBoxes failed");
      }
    }

    // Frame settings (owned by encoder, no manual cleanup needed)
    JxlEncoderFrameSettings *frame_settings = JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
    JxlEncoderFrameSettingsSetOption(frame_settings, JXL_ENC_FRAME_SETTING_EFFORT, effort);
    if (lossless) {
      JxlEncoderSetFrameLossless(frame_settings, JXL_TRUE);
    } else {
      JxlEncoderSetFrameDistance(frame_settings, distance);
    }

    // Basic info
    JxlBasicInfo basic_info;
    JxlEncoderInitBasicInfo(&basic_info);
    basic_info.xsize = static_cast<uint32_t>(width);
    basic_info.ysize = static_cast<uint32_t>(height);
    basic_info.bits_per_sample = 8;
    basic_info.uses_original_profile = JXL_TRUE;
    if (channels == 4) {
      basic_info.num_extra_channels = 1;
      basic_info.alpha_bits = 8;
    }

    if (JXL_ENC_SUCCESS != JxlEncoderSetBasicInfo(enc.get(), &basic_info)) {
      throw std::runtime_error("JxlEncoderSetBasicInfo failed: " +
                               std::to_string(JxlEncoderGetError(enc.get())));
    }

    // Color encoding
    JxlColorEncoding color_encoding = {};
    JxlColorEncodingSetToSRGB(&color_encoding, JXL_FALSE);
    if (JXL_ENC_SUCCESS != JxlEncoderSetColorEncoding(enc.get(), &color_encoding)) {
      throw std::runtime_error("JxlEncoderSetColorEncoding failed: " +
                               std::to_string(JxlEncoderGetError(enc.get())));
    }

    // Add image frame
    JxlPixelFormat pixel_format = {
        static_cast<uint32_t>(channels), JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0};

    if (JXL_ENC_SUCCESS !=
        JxlEncoderAddImageFrame(frame_settings, &pixel_format, input_ptr, input_size)) {
      throw std::runtime_error("JxlEncoderAddImageFrame failed: " +
                               std::to_string(JxlEncoderGetError(enc.get())));
    }

    // ── Add metadata boxes ──
    if (has_metadata) {
      JxlEncoderCloseFrames(enc.get());

      if (!exif_data.empty()) {
        // EXIF box requires 4-byte TIFF header offset prefix (usually 0)
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
          throw std::runtime_error("JxlEncoderAddBox(xml) failed");
        }
      }

      if (!jumbf_data.empty()) {
        if (JXL_ENC_SUCCESS !=
            JxlEncoderAddBox(enc.get(), "jumb", jumbf_data.data(), jumbf_data.size(), JXL_TRUE)) {
          throw std::runtime_error("JxlEncoderAddBox(jumb) failed");
        }
      }

      JxlEncoderCloseBoxes(enc.get());
    } else {
      JxlEncoderCloseInput(enc.get());
    }

    // Process output — pre-estimate buffer size to reduce reallocations
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
    compressed.shrink_to_fit();
  }
  // GIL is re-acquired here by RAII destructor

  return py::bytes(reinterpret_cast<const char *>(compressed.data()), compressed.size());
}

// ─── Decode
// ────────────────────────────────────────────────────────────────────

// NOLINTNEXTLINE(readability-function-cognitive-complexity,misc-use-internal-linkage)
py::object decode(py::bytes data, bool metadata = false) {
  // ── Extract pointer without copying (GIL held) ──
  char *raw_ptr = nullptr; // NOLINT(misc-const-correctness) — PyBytes API requires non-const
  Py_ssize_t raw_size = 0;
  if (PyBytes_AsStringAndSize(data.ptr(), &raw_ptr, &raw_size) != 0) {
    throw py::error_already_set();
  }
  const auto *jxl_data = reinterpret_cast<const uint8_t *>(raw_ptr);
  const auto jxl_size = static_cast<size_t>(raw_size);

  // ── Pass 1: Read BasicInfo (GIL released) ──
  JxlBasicInfo info;
  {
    py::gil_scoped_release release;

    JxlDecoderPtr dec(JxlDecoderCreate(nullptr));
    if (!dec) {
      throw std::runtime_error("JxlDecoderCreate failed");
    }

    if (JXL_DEC_SUCCESS != JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_BASIC_INFO)) {
      throw std::runtime_error("JxlDecoderSubscribeEvents failed");
    }

    JxlDecoderSetInput(dec.get(), jxl_data, jxl_size);
    JxlDecoderCloseInput(dec.get());

    for (;;) {
      JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());
      if (status == JXL_DEC_ERROR) {
        throw std::runtime_error("Decoder error while reading header");
      }
      if (status == JXL_DEC_NEED_MORE_INPUT) {
        throw std::runtime_error("Truncated JXL data: need more input for header");
      }
      if (status == JXL_DEC_BASIC_INFO) {
        if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(dec.get(), &info)) {
          throw std::runtime_error("JxlDecoderGetBasicInfo failed");
        }
        break;
      }
      if (status == JXL_DEC_SUCCESS) {
        throw std::runtime_error("Decoder finished without providing BasicInfo");
      }
    }
  }
  // GIL is re-acquired here

  // ── Allocate output array (GIL held — safe for Python allocation) ──
  const size_t channels = info.num_color_channels + (info.alpha_bits > 0 ? 1 : 0);
  py::array_t<uint8_t> result(
      {static_cast<size_t>(info.ysize), static_cast<size_t>(info.xsize), channels});
  auto result_buf = result.request();
  auto *result_ptr = result_buf.ptr;
  const auto result_bytes = static_cast<size_t>(result_buf.size * result_buf.itemsize);

  // ── Pass 2: Decode pixels + optional metadata (GIL released) ──
  std::map<std::string, std::vector<uint8_t>> boxes;
  {
    py::gil_scoped_release release;

    JxlRunnerPtr runner(JxlResizableParallelRunnerCreate(nullptr));
    if (!runner) {
      throw std::runtime_error("JxlResizableParallelRunnerCreate failed");
    }
    JxlResizableParallelRunnerSetThreads(
        runner.get(), JxlResizableParallelRunnerSuggestThreads(info.xsize, info.ysize));

    JxlDecoderPtr dec(JxlDecoderCreate(nullptr));
    if (!dec) {
      throw std::runtime_error("JxlDecoderCreate failed");
    }

    if (JXL_DEC_SUCCESS !=
        JxlDecoderSetParallelRunner(dec.get(), JxlResizableParallelRunner, runner.get())) {
      throw std::runtime_error("JxlDecoderSetParallelRunner failed");
    }

    int events = JXL_DEC_FULL_IMAGE;
    if (metadata) {
      events |= JXL_DEC_BOX;
      JxlDecoderSetDecompressBoxes(dec.get(), JXL_TRUE);
    }
    if (JXL_DEC_SUCCESS != JxlDecoderSubscribeEvents(dec.get(), events)) {
      throw std::runtime_error("JxlDecoderSubscribeEvents failed");
    }

    JxlDecoderSetInput(dec.get(), jxl_data, jxl_size);
    JxlDecoderCloseInput(dec.get());

    JxlPixelFormat format = {static_cast<uint32_t>(channels), JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0};

    // Box reading state
    std::string current_box_type;
    std::vector<uint8_t> box_buffer;
    constexpr size_t kBoxChunkSize = 65536;

    for (;;) {
      JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());

      if (status == JXL_DEC_ERROR) {
        throw std::runtime_error("Decoder error during pixel decode");
      }
      if (status == JXL_DEC_NEED_MORE_INPUT) {
        throw std::runtime_error("Truncated JXL data: need more input for pixels");
      }
      if (status == JXL_DEC_BASIC_INFO) {
        // Skip — we already have info from pass 1
        continue;
      }
      if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
        if (JXL_DEC_SUCCESS !=
            JxlDecoderSetImageOutBuffer(dec.get(), &format, result_ptr, result_bytes)) {
          throw std::runtime_error("JxlDecoderSetImageOutBuffer failed");
        }
        continue;
      }
      if (status == JXL_DEC_BOX) {
        // Finish previous box if any
        if (!current_box_type.empty()) {
          size_t remaining = JxlDecoderReleaseBoxBuffer(dec.get());
          box_buffer.resize(box_buffer.size() - remaining);
          boxes[current_box_type] = std::move(box_buffer);
          current_box_type.clear();
        }

        // Read new box type (decompressed)
        JxlBoxType box_type{};
        if (JXL_DEC_SUCCESS != JxlDecoderGetBoxType(dec.get(), box_type, JXL_TRUE)) {
          continue;
        }
        std::string type_str(box_type, 4);

        // Only capture metadata boxes
        if (type_str == "Exif" || type_str == "xml " || type_str == "jumb") {
          current_box_type = type_str;
          box_buffer.resize(kBoxChunkSize);
          JxlDecoderSetBoxBuffer(dec.get(), box_buffer.data(), box_buffer.size());
        }
        continue;
      }
      if (status == JXL_DEC_BOX_NEED_MORE_OUTPUT) {
        // Grow box buffer
        size_t remaining = JxlDecoderReleaseBoxBuffer(dec.get());
        size_t bytes_read = box_buffer.size() - remaining;
        box_buffer.resize(box_buffer.size() + kBoxChunkSize);
        JxlDecoderSetBoxBuffer(
            dec.get(), box_buffer.data() + bytes_read, box_buffer.size() - bytes_read);
        continue;
      }
      if (status == JXL_DEC_FULL_IMAGE) {
        if (!metadata) {
          break;
        }
        continue; // keep going to read remaining boxes
      }
      if (status == JXL_DEC_SUCCESS) {
        // Finish last box if any
        if (!current_box_type.empty()) {
          size_t remaining = JxlDecoderReleaseBoxBuffer(dec.get());
          box_buffer.resize(box_buffer.size() - remaining);
          boxes[current_box_type] = std::move(box_buffer);
        }
        break;
      }
    }
  }
  // GIL is re-acquired here

  if (!metadata) {
    return result;
  }

  // Build metadata dict
  py::dict meta;
  for (auto &[key, value] : boxes) {
    if (key == "Exif" && value.size() > 4) {
      // Strip the 4-byte TIFF header offset prefix
      meta[py::cast("exif")] =
          py::bytes(reinterpret_cast<const char *>(value.data() + 4), value.size() - 4);
    } else if (key == "xml ") {
      meta[py::cast("xmp")] = py::bytes(reinterpret_cast<const char *>(value.data()), value.size());
    } else if (key == "jumb") {
      meta[py::cast("jumbf")] =
          py::bytes(reinterpret_cast<const char *>(value.data()), value.size());
    }
  }
  return py::make_tuple(result, meta);
}

// ─── JPEG Codec (TurboJPEG)
// ──────────────────────────────────────────────────────────

// NOLINTNEXTLINE(misc-use-internal-linkage)
py::bytes encode_jpeg(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> input,
                      int quality = 95) {
  auto buf = input.request();
  if (buf.ndim != 3) {
    throw std::invalid_argument("Input must be a 3D array (height, width, channels)");
  }
  const auto height = static_cast<int>(buf.shape[0]);
  const auto width = static_cast<int>(buf.shape[1]);
  const auto channels = static_cast<int>(buf.shape[2]);

  if (channels != 3 && channels != 4) {
    throw std::invalid_argument("Input must have 3 (RGB) or 4 (RGBA) channels");
  }

  quality = std::clamp(quality, 1, 100);

  unsigned char *jpeg_buf = nullptr;
  unsigned long jpeg_size = 0; // NOLINT(google-runtime-int)

  {
    py::gil_scoped_release release;

    TjPtr compressor(tjInitCompress());
    if (!compressor) {
      throw std::runtime_error("tjInitCompress failed");
    }

    int pixel_format = (channels == 3) ? TJPF_RGB : TJPF_RGBA;
    int subsamp = (channels == 3)
                      ? TJSAMP_444
                      : TJSAMP_444; // For RGBA, alpha is dropped, so treat as RGB for subsampling
    // tjCompress2 takes output subsampling.
    // If input is RGBA, we can compress to YCbCr (RGB) and drop alpha.

    if (tjCompress2(compressor.get(),
                    static_cast<const unsigned char *>(buf.ptr),
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
  // GIL re-acquired

  TjBufPtr guard(jpeg_buf); // Ensures tjFree is called
  return py::bytes(reinterpret_cast<const char *>(jpeg_buf), jpeg_size);
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
py::array_t<uint8_t> decode_jpeg(py::bytes data) {
  char *raw_ptr = nullptr; // NOLINT
  Py_ssize_t raw_size = 0;
  if (PyBytes_AsStringAndSize(data.ptr(), &raw_ptr, &raw_size) != 0) {
    throw py::error_already_set();
  }

  const auto *jpeg_data = reinterpret_cast<const unsigned char *>(raw_ptr);
  const auto jpeg_size = static_cast<unsigned long>(raw_size); // NOLINT

  int width = 0, height = 0, subsamp = 0, colorspace = 0;
  {
    py::gil_scoped_release release;
    TjPtr decompressor(tjInitDecompress());
    if (!decompressor) {
      throw std::runtime_error("tjInitDecompress failed");
    }
    if (tjDecompressHeader3(
            decompressor.get(), jpeg_data, jpeg_size, &width, &height, &subsamp, &colorspace) !=
        0) {
      throw std::runtime_error(std::string("tjDecompressHeader3 failed: ") +
                               tjGetErrorStr2(decompressor.get()));
    }
  }

  // Allocate result (GIL held)
  // We'll output RGB (3 channels) by default.
  py::array_t<uint8_t> result({height, width, 3});
  auto buf = result.request();

  {
    py::gil_scoped_release release;
    TjPtr decompressor(tjInitDecompress());
    if (tjDecompress2(decompressor.get(),
                      jpeg_data,
                      jpeg_size,
                      static_cast<unsigned char *>(buf.ptr),
                      width,
                      0,
                      height,
                      TJPF_RGB,
                      TJFLAG_FASTDCT) != 0) {
      throw std::runtime_error(std::string("tjDecompress2 failed: ") +
                               tjGetErrorStr2(decompressor.get()));
    }
  }

  return result;
}

// ─── JXL <-> JPEG Transcoding
// ──────────────────────────────────────────────────────────

// NOLINTNEXTLINE(misc-use-internal-linkage)
py::bytes jpeg_to_jxl(py::bytes jpeg_data, int effort = 7) {
  char *raw_ptr = nullptr; // NOLINT
  Py_ssize_t raw_size = 0;
  if (PyBytes_AsStringAndSize(jpeg_data.ptr(), &raw_ptr, &raw_size) != 0) {
    throw py::error_already_set();
  }
  const auto *jpeg_ptr = reinterpret_cast<const uint8_t *>(raw_ptr);
  const auto jpeg_len = static_cast<size_t>(raw_size);

  effort = std::clamp(effort, 1, 10);

  std::vector<uint8_t> compressed;
  {
    py::gil_scoped_release release;

    JxlEncoderPtr enc(JxlEncoderCreate(nullptr));
    if (!enc)
      throw std::runtime_error("JxlEncoderCreate failed");

    // Use container to allow metadata (exif/xmp) preservation from JPEG
    if (JXL_ENC_SUCCESS != JxlEncoderUseContainer(enc.get())) {
      throw std::runtime_error("JxlEncoderUseContainer failed");
    }

    // Frame settings
    JxlEncoderFrameSettings *settings = JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
    if (JXL_ENC_SUCCESS !=
        JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, effort)) {
      throw std::runtime_error("JxlEncoderFrameSettingsSetOption(EFFORT) failed");
    }

    // Add JPEG frame (handles recompression and metadata)
    if (JXL_ENC_SUCCESS != JxlEncoderAddJPEGFrame(settings, jpeg_ptr, jpeg_len)) {
      throw std::runtime_error("JxlEncoderAddJPEGFrame failed (input may not be a valid JPEG)");
    }

    JxlEncoderCloseInput(enc.get());

    // Process output
    compressed.resize(jpeg_len + 4096); // heuristic initial size
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
  return py::bytes(reinterpret_cast<const char *>(compressed.data()), compressed.size());
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
py::bytes jxl_to_jpeg(py::bytes jxl_data) {
  char *raw_ptr = nullptr; // NOLINT
  Py_ssize_t raw_size = 0;
  if (PyBytes_AsStringAndSize(jxl_data.ptr(), &raw_ptr, &raw_size) != 0) {
    throw py::error_already_set();
  }
  const auto *jxl_ptr = reinterpret_cast<const uint8_t *>(raw_ptr);
  const auto jxl_len = static_cast<size_t>(raw_size);

  std::vector<uint8_t> jpeg_out;
  {
    py::gil_scoped_release release;
    JxlDecoderPtr dec(JxlDecoderCreate(nullptr));
    if (!dec)
      throw std::runtime_error("JxlDecoderCreate failed");

    if (JXL_DEC_SUCCESS != JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_JPEG_RECONSTRUCTION)) {
      throw std::runtime_error("JxlDecoderSubscribeEvents failed");
    }

    JxlDecoderSetInput(dec.get(), jxl_ptr, jxl_len);
    JxlDecoderCloseInput(dec.get());

    std::vector<uint8_t> buffer(jxl_len * 2 + 16384); // Initial guess
    bool reconstruction_found = false;

    for (;;) {
      JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());
      if (status == JXL_DEC_ERROR) {
        throw std::runtime_error("Decoder error");
      }
      if (status == JXL_DEC_JPEG_RECONSTRUCTION) {
        reconstruction_found = true;
        // Provide buffer
        if (JXL_DEC_SUCCESS != JxlDecoderSetJPEGBuffer(dec.get(), buffer.data(), buffer.size())) {
          throw std::runtime_error("JxlDecoderSetJPEGBuffer failed");
        }
      } else if (status == JXL_DEC_SUCCESS) {
        size_t remaining = JxlDecoderReleaseJPEGBuffer(dec.get());
        size_t used = buffer.size() - remaining;
        jpeg_out.assign(buffer.data(), buffer.data() + used);
        break;
      } else if (status == JXL_DEC_NEED_MORE_INPUT) {
        throw std::runtime_error("Truncated JXL data");
      } else if (status == JXL_DEC_FULL_IMAGE) {
        // If we reached full image without JPEG reconstruction, it means it wasn't a recompressed
        // JPEG
        break; // reconstruction_found will be false (or maybe true but finished?)
      }
    }

    if (!reconstruction_found || jpeg_out.empty()) {
      throw std::runtime_error("JXL data does not contain a reconstructible JPEG codestream");
    }
  }
  return py::bytes(reinterpret_cast<const char *>(jpeg_out.data()), jpeg_out.size());
}

// ─── Unified Codec Context Manager
// ──────────────────────────────────────────

class PyJxlCodec {
public:
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters,cppcoreguidelines-pro-type-member-init)
  PyJxlCodec(int effort = 7, float distance = 1.0F, bool lossless = false)
      : effort_(std::clamp(effort, 1, 10)),
        distance_(lossless ? 0.0F : std::clamp(distance, 0.0F, 25.0F)), lossless_(lossless) {}

  /// Encode an image, with optional per-call overrides and metadata.
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  py::bytes encode_image(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> input,
                         std::optional<int> effort,
                         std::optional<float> distance,
                         std::optional<bool> lossless,
                         py::object exif,
                         py::object xmp,
                         py::object jumbf) {
    check_closed();
    int eff = effort.value_or(effort_);
    bool ll = lossless.value_or(lossless_);
    float dist = distance.value_or(ll ? 0.0F : distance_);
    return encode(input, eff, dist, ll, exif, xmp, jumbf);
  }

  /// Decode JXL bytes, optionally extracting metadata.
  py::object decode_image(py::bytes data, bool metadata) {
    check_closed();
    return decode(data, metadata);
  }

  PyJxlCodec &enter() {
    check_closed();
    return *this;
  }

  void exit(const py::object & /*exc_type*/,
            const py::object & /*exc_val*/,
            const py::object & /*exc_tb*/) {
    close();
  }

  void close() { closed_ = true; }
  [[nodiscard]] bool closed() const { return closed_; }

private:
  void check_closed() const {
    if (closed_) {
      throw std::runtime_error("Cannot use a closed JXL codec");
    }
  }

  int effort_;
  float distance_;
  bool lossless_;
  bool closed_ = false;
};

// ─── Module Definition
// ─────────────────────────────────────────────────────────

PYBIND11_MODULE(_pylibjxl, m) { // NOLINT
  m.doc() = "Python bindings for libjxl";

  m.def(
      "version",
      []() {
        return py::dict("major"_a = JPEGXL_MAJOR_VERSION,
                        "minor"_a = JPEGXL_MINOR_VERSION,
                        "patch"_a = JPEGXL_PATCH_VERSION);
      },
      "Get libjxl version");

  m.def("decoder_version", &JxlDecoderVersion, "Get libjxl decoder version");
  m.def("encoder_version", &JxlEncoderVersion, "Get libjxl encoder version");

  m.def("encode",
        &encode,
        "Encode a numpy array (H, W, C) to JXL bytes.\n\n"
        "Args:\n"
        "    input: uint8 numpy array of shape (height, width, channels)\n"
        "    effort: Encoding effort [1-10], higher = slower + smaller (default 7)\n"
        "    distance: Perceptual distance [0.0-25.0], 0 = lossless (default 1.0)\n"
        "    lossless: If True, encode losslessly (default False)\n"
        "    exif: Optional EXIF metadata as bytes\n"
        "    xmp: Optional XMP metadata as bytes\n"
        "    jumbf: Optional JUMBF metadata as bytes\n",
        "input"_a,
        "effort"_a = 7,
        "distance"_a = 1.0F,
        "lossless"_a = false,
        "exif"_a = py::none(),
        "xmp"_a = py::none(),
        "jumbf"_a = py::none());

  m.def("decode",
        &decode,
        "Decode JXL bytes to a uint8 numpy array (H, W, C).\n\n"
        "When metadata=True, returns a tuple of (array, dict) where dict\n"
        "contains the extracted metadata (exif, xmp, jumbf as bytes).\n\n"
        "Args:\n"
        "    data: bytes object containing JXL-encoded data\n"
        "    metadata: If True, also extract metadata boxes (default False)\n",
        "data"_a,
        "metadata"_a = false);

  // ── Unified JXL codec context manager ──
  py::class_<PyJxlCodec>(m,
                         "JXL",
                         "Unified JXL codec with context manager support.\n\n"
                         "Provides both encode() and decode() in a single context.\n\n"
                         "Usage:\n"
                         "    with pylibjxl.JXL(effort=7) as jxl:\n"
                         "        data = jxl.encode(image)\n"
                         "        image = jxl.decode(data)\n")
      .def(py::init<int, float, bool>(), "effort"_a = 7, "distance"_a = 1.0F, "lossless"_a = false)
      .def("encode",
           &PyJxlCodec::encode_image,
           "Encode a numpy array to JXL bytes.\n\n"
           "Per-call overrides take precedence over constructor defaults.",
           "input"_a,
           "effort"_a = py::none(),
           "distance"_a = py::none(),
           "lossless"_a = py::none(),
           "exif"_a = py::none(),
           "xmp"_a = py::none(),
           "jumbf"_a = py::none())
      .def("decode",
           &PyJxlCodec::decode_image,
           "Decode JXL bytes, optionally extracting metadata.",
           "data"_a,
           "metadata"_a = false)
      .def("close", &PyJxlCodec::close, "Close the codec, preventing further use.")
      .def_property_readonly("closed", &PyJxlCodec::closed, "Whether the codec has been closed.")
      .def("__enter__", &PyJxlCodec::enter, py::return_value_policy::reference)
      .def("__exit__", &PyJxlCodec::exit);

  // ─── JPEG & Transcoding ───
  m.def("encode_jpeg",
        &encode_jpeg,
        "Encode numpy array to JPEG bytes (using libjpeg-turbo).\n"
        "Input: (H, W, 3) or (H, W, 4).\n"
        "Quality: 1-100 (default 95).",
        "input"_a,
        "quality"_a = 95);

  m.def("decode_jpeg",
        &decode_jpeg,
        "Decode JPEG bytes to numpy array (H, W, 3) (using libjpeg-turbo).",
        "data"_a);

  m.def("jpeg_to_jxl",
        &jpeg_to_jxl,
        "Losslessly recompress valid JPEG bytes to JXL bytes.",
        "data"_a,
        "effort"_a = 7);

  m.def("jxl_to_jpeg",
        &jxl_to_jpeg,
        "Reconstruct original JPEG bytes from JXL bytes (if recompressed).",
        "data"_a);
}
