#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/version.h>

#include "bindings/py_codec.hpp"
#include "codecs/jpeg_ops.hpp"
#include "codecs/jxl_ops.hpp"
#include "codecs/transcode.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace pylibjxl;

NB_MODULE(_pylibjxl, m) {
  m.doc() = "Python bindings for libjxl";

  nb::exception<CodecTimeoutError>(m, "CodecTimeoutError", PyExc_TimeoutError);

  m.def(
      "version",
      []() {
        nb::dict d;
        d["major"] = JPEGXL_MAJOR_VERSION;
        d["minor"] = JPEGXL_MINOR_VERSION;
        d["patch"] = JPEGXL_PATCH_VERSION;
        return d;
      },
      "Get libjxl version");

  m.def("decoder_version", &JxlDecoderVersion, "Get libjxl decoder version");
  m.def("encoder_version", &JxlEncoderVersion, "Get libjxl encoder version");

  m.def(
      "encode",
      [](nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu> input,
         int effort,
         float distance,
         bool lossless,
         int decoding_speed,
         nb::handle exif,
         nb::handle xmp,
         nb::handle jumbf,
         nb::handle icc,
         std::optional<double> timeout) {
        std::optional<std::chrono::milliseconds> t;
        if (timeout.has_value()) {
          t = std::chrono::milliseconds(static_cast<int64_t>(*timeout * 1000.0));
        }
        return encode(input, effort, distance, lossless, decoding_speed, exif, xmp, jumbf, icc, t);
      },
      "Encode a numpy array (H, W, C) to JXL bytes.\n\n"
      "Args:\n"
      "    input: uint8 numpy array of shape (height, width, channels)\n"
      "    effort: Encoding effort [1-11], higher = slower + smaller (default 7)\n"
      "    distance: Perceptual distance [0.0-25.0], 0 = lossless (default 1.0)\n"
      "    lossless: If True, encode losslessly (default False)\n"
      "    decoding_speed: Decoding speed tier [0-4], higher = faster to decode (default 0)\n"
      "    exif: Optional EXIF metadata as bytes\n"
      "    xmp: Optional XMP metadata as bytes\n"
      "    jumbf: Optional JUMBF metadata as bytes\n"
      "    icc: Optional ICC profile metadata as bytes\n"
      "    timeout: Optional acquisition timeout in seconds\n",
      "input"_a,
      "effort"_a = 7,
      "distance"_a = 1.0F,
      "lossless"_a = false,
      "decoding_speed"_a = 0,
      "exif"_a = nb::none(),
      "xmp"_a = nb::none(),
      "jumbf"_a = nb::none(),
      "icc"_a = nb::none(),
      "timeout"_a = nb::none());

  m.def(
      "decode",
      [](nb::handle data,
         bool metadata,
         std::optional<nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu>> out,
         std::optional<double> timeout) {
        std::optional<std::chrono::milliseconds> t;
        if (timeout.has_value()) {
          t = std::chrono::milliseconds(static_cast<int64_t>(*timeout * 1000.0));
        }
        return decode(data, metadata, out, t);
      },
      "Decode JXL data (bytes, bytearray, memoryview, mmap) to a uint8 numpy array (H, W, C).\n\n"
      "When metadata=True, returns a tuple of (array, dict) where dict\n"
      "contains the extracted metadata (exif, xmp, jumbf, icc, icc_profile as bytes).\n\n"
      "When out is provided, decodes in-place into the pre-allocated C-contiguous array.\n\n"
      "Args:\n"
      "    data: Buffer object containing JXL-encoded data\n"
      "    metadata: If True, also extract metadata boxes and color profiles (default False)\n"
      "    out: Optional pre-allocated uint8 numpy array (H, W, C) for zero-copy in-place decode\n"
      "    timeout: Optional acquisition timeout in seconds\n",
      "data"_a,
      "metadata"_a = false,
      "out"_a = nb::none(),
      "timeout"_a = nb::none());

  m.def(
      "probe",
      &probe,
      "Probe JXL header metadata (dimensions, channels, suggested threads) in <0.05ms.\n\n"
      "Zero runner pool contention: executes directly on the caller thread without acquiring workers.\n\n"
      "Args:\n"
      "    data: Buffer object containing JXL-encoded data\n"
      "Returns:\n"
      "    dict containing width, height, channels, color_channels, has_alpha, bits_per_sample,\n"
      "    exponent_bits_per_sample, have_animation, suggested_threads\n",
      "data"_a);

  nb::class_<PyJxlCodec>(m,
                         "JXL",
                         "Unified JXL/JPEG codec with context manager support.\n\n"
                         "Owns an elastic thread pool that is destroyed on close().\n"
                         "Supports JXL encode/decode, JPEG encode/decode, and\n"
                         "cross-format transcoding.\n\n"
                         "Usage:\n"
                         "    with pylibjxl.JXL(effort=7, pool_size=4, timeout=5.0) as jxl:\n"
                         "        data = jxl.encode(image)\n"
                         "        image = jxl.decode(data)\n"
                         "        jpeg = jxl.encode_jpeg(image)\n"
                         "        img = jxl.decode_jpeg(jpeg)\n"
                         "        jxl_data = jxl.jpeg_to_jxl(jpeg)\n"
                         "        jpeg_back = jxl.jxl_to_jpeg(jxl_data)\n")
      .def(nb::init<int, float, bool, int, int, int, std::optional<double>, double>(),
           "effort"_a = 7,
           "distance"_a = 1.0F,
           "lossless"_a = false,
           "decoding_speed"_a = 0,
           "threads"_a = 0,
           "pool_size"_a = 0,
           "timeout"_a = nb::none(),
           "idle_timeout"_a = 30.0)
      .def("encode",
           &PyJxlCodec::encode_image,
           "Encode a numpy array to JXL bytes.\n\n"
           "Per-call overrides take precedence over constructor defaults.",
           "input"_a,
           "effort"_a = nb::none(),
           "distance"_a = nb::none(),
           "lossless"_a = nb::none(),
           "decoding_speed"_a = nb::none(),
           "exif"_a = nb::none(),
           "xmp"_a = nb::none(),
           "jumbf"_a = nb::none(),
           "icc"_a = nb::none(),
           "timeout"_a = nb::none())
      .def("decode",
           &PyJxlCodec::decode_image,
           "Decode JXL bytes, optionally extracting metadata and using an in-place output buffer.",
           "data"_a,
           "metadata"_a = false,
           "out"_a = nb::none(),
           "timeout"_a = nb::none())
      .def("probe",
           &PyJxlCodec::probe_image,
           "Probe JXL header metadata (dimensions, channels, suggested threads) with zero contention.",
           "data"_a)
      .def("encode_jpeg",
           &PyJxlCodec::encode_jpeg_image,
           "Encode numpy array to JPEG bytes (uses libjpeg-turbo, zero-copy direct output).",
           "input"_a,
           "quality"_a = 95)
      .def(
          "decode_jpeg",
          &PyJxlCodec::decode_jpeg_image,
          "Decode JPEG bytes to numpy array (H, W, 3), optionally using an in-place output buffer.",
          "data"_a,
          "out"_a = nb::none())
      .def("jpeg_to_jxl",
           &PyJxlCodec::jpeg_to_jxl_image,
           "Losslessly recompress JPEG bytes to JXL bytes.",
           "data"_a,
           "effort"_a = nb::none(),
           "timeout"_a = nb::none())
      .def("jxl_to_jpeg",
           &PyJxlCodec::jxl_to_jpeg_image,
           "Reconstruct original JPEG bytes from JXL bytes.",
           "data"_a,
           "timeout"_a = nb::none())
      .def("jpeg_to_jxl_file",
           &PyJxlCodec::jpeg_to_jxl_file_image,
           "Direct C++ file-to-file lossless JPEG to JXL recompression.",
           "in_path"_a,
           "out_path"_a,
           "effort"_a = nb::none(),
           "timeout"_a = nb::none())
      .def("jxl_to_jpeg_file",
           &PyJxlCodec::jxl_to_jpeg_file_image,
           "Direct C++ file-to-file lossless JXL to JPEG reconstruction.",
           "in_path"_a,
           "out_path"_a,
           "timeout"_a = nb::none())
      .def("close", &PyJxlCodec::close, "Close the codec and release thread pool resources.")
      .def_prop_ro("closed", &PyJxlCodec::closed, "Whether the codec has been closed.")
      .def_prop_ro("pool_size", &PyJxlCodec::pool_size, "Maximum number of parallel runners in the pool.")
      .def_prop_ro("total_runners", &PyJxlCodec::total_runners, "Total runners currently allocated.")
      .def_prop_ro("available_runners", &PyJxlCodec::available_runners, "Idle runners ready for immediate use.")
      .def_prop_ro("in_use_runners", &PyJxlCodec::in_use_runners, "Number of runners currently borrowed/active.")
      .def_prop_ro("threads_per_runner", &PyJxlCodec::threads_per_runner, "Worker threads allocated to each runner.")
      .def("__enter__", &PyJxlCodec::enter, nb::rv_policy::reference)
      .def("__exit__",
           &PyJxlCodec::exit,
           nb::arg("exc_type") = nb::none(),
           nb::arg("exc_val") = nb::none(),
           nb::arg("exc_tb") = nb::none());

  m.def("encode_jpeg",
        &encode_jpeg,
        "Encode numpy array to JPEG bytes (using libjpeg-turbo with zero intermediate copies).\n"
        "Input: (H, W, 3) or (H, W, 4).\n"
        "Quality: 1-100 (default 95).",
        "input"_a,
        "quality"_a = 95);

  m.def("decode_jpeg",
        &decode_jpeg,
        "Decode JPEG bytes to numpy array (H, W, 3) (using libjpeg-turbo).\n"
        "When out is provided, decodes in-place into the pre-allocated C-contiguous array.\n",
        "data"_a,
        "out"_a = nb::none());

  m.def(
      "jpeg_to_jxl",
      [](nb::handle data, int effort, std::optional<double> timeout) {
        std::optional<std::chrono::milliseconds> t;
        if (timeout.has_value()) {
          t = std::chrono::milliseconds(static_cast<int64_t>(*timeout * 1000.0));
        }
        return jpeg_to_jxl(data, effort, t);
      },
      "Losslessly recompress valid JPEG data to JXL bytes.",
      "data"_a,
      "effort"_a = 7,
      "timeout"_a = nb::none());

  m.def(
      "jxl_to_jpeg",
      [](nb::handle data, std::optional<double> timeout) {
        std::optional<std::chrono::milliseconds> t;
        if (timeout.has_value()) {
          t = std::chrono::milliseconds(static_cast<int64_t>(*timeout * 1000.0));
        }
        return jxl_to_jpeg(data, t);
      },
      "Reconstruct original JPEG data from JXL bytes (if recompressed).",
      "data"_a,
      "timeout"_a = nb::none());

  m.def(
      "jpeg_to_jxl_file",
      [](const std::string &in_path,
         const std::string &out_path,
         int effort,
         std::optional<double> timeout) {
        std::optional<std::chrono::milliseconds> t;
        if (timeout.has_value()) {
          t = std::chrono::milliseconds(static_cast<int64_t>(*timeout * 1000.0));
        }
        jpeg_to_jxl_file(in_path, out_path, effort, t);
      },
      "Direct C++ file-to-file lossless JPEG to JXL recompression (zero Python heap overhead).",
      "in_path"_a,
      "out_path"_a,
      "effort"_a = 7,
      "timeout"_a = nb::none());

  m.def(
      "jxl_to_jpeg_file",
      [](const std::string &in_path,
         const std::string &out_path,
         std::optional<double> timeout) {
        std::optional<std::chrono::milliseconds> t;
        if (timeout.has_value()) {
          t = std::chrono::milliseconds(static_cast<int64_t>(*timeout * 1000.0));
        }
        jxl_to_jpeg_file(in_path, out_path, t);
      },
      "Direct C++ file-to-file lossless JXL to JPEG reconstruction (zero Python heap overhead).",
      "in_path"_a,
      "out_path"_a,
      "timeout"_a = nb::none());
}
