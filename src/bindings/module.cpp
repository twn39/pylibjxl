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

  m.def("encode",
        &encode,
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
        "    icc: Optional ICC profile metadata as bytes\n",
        "input"_a,
        "effort"_a = 7,
        "distance"_a = 1.0F,
        "lossless"_a = false,
        "decoding_speed"_a = 0,
        "exif"_a = nb::none(),
        "xmp"_a = nb::none(),
        "jumbf"_a = nb::none(),
        "icc"_a = nb::none());

  m.def(
      "decode",
      &decode,
      "Decode JXL data (bytes, bytearray, memoryview, mmap) to a uint8 numpy array (H, W, C).\n\n"
      "When metadata=True, returns a tuple of (array, dict) where dict\n"
      "contains the extracted metadata (exif, xmp, jumbf, icc, icc_profile as bytes).\n\n"
      "When out is provided, decodes in-place into the pre-allocated C-contiguous array.\n\n"
      "Args:\n"
      "    data: Buffer object containing JXL-encoded data\n"
      "    metadata: If True, also extract metadata boxes and color profiles (default False)\n"
      "    out: Optional pre-allocated uint8 numpy array (H, W, C) for zero-copy in-place decode\n",
      "data"_a,
      "metadata"_a = false,
      "out"_a = nb::none());

  nb::class_<PyJxlCodec>(m,
                         "JXL",
                         "Unified JXL/JPEG codec with context manager support.\n\n"
                         "Owns a shared thread pool that is destroyed on close().\n"
                         "Supports JXL encode/decode, JPEG encode/decode, and\n"
                         "cross-format transcoding.\n\n"
                         "Usage:\n"
                         "    with pylibjxl.JXL(effort=7) as jxl:\n"
                         "        data = jxl.encode(image)\n"
                         "        image = jxl.decode(data)\n"
                         "        jpeg = jxl.encode_jpeg(image)\n"
                         "        img = jxl.decode_jpeg(jpeg)\n"
                         "        jxl_data = jxl.jpeg_to_jxl(jpeg)\n"
                         "        jpeg_back = jxl.jxl_to_jpeg(jxl_data)\n")
      .def(nb::init<int, float, bool, int, int>(),
           "effort"_a = 7,
           "distance"_a = 1.0F,
           "lossless"_a = false,
           "decoding_speed"_a = 0,
           "threads"_a = 0)
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
           "icc"_a = nb::none())
      .def("decode",
           &PyJxlCodec::decode_image,
           "Decode JXL bytes, optionally extracting metadata and using an in-place output buffer.",
           "data"_a,
           "metadata"_a = false,
           "out"_a = nb::none())
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
           "effort"_a = nb::none())
      .def("jxl_to_jpeg",
           &PyJxlCodec::jxl_to_jpeg_image,
           "Reconstruct original JPEG bytes from JXL bytes.",
           "data"_a)
      .def("jpeg_to_jxl_file",
           &PyJxlCodec::jpeg_to_jxl_file_image,
           "Direct C++ file-to-file lossless JPEG to JXL recompression.",
           "in_path"_a,
           "out_path"_a,
           "effort"_a = nb::none())
      .def("jxl_to_jpeg_file",
           &PyJxlCodec::jxl_to_jpeg_file_image,
           "Direct C++ file-to-file lossless JXL to JPEG reconstruction.",
           "in_path"_a,
           "out_path"_a)
      .def("close", &PyJxlCodec::close, "Close the codec and release thread pool resources.")
      .def_prop_ro("closed", &PyJxlCodec::closed, "Whether the codec has been closed.")
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

  m.def("jpeg_to_jxl",
        &jpeg_to_jxl,
        "Losslessly recompress valid JPEG data to JXL bytes.",
        "data"_a,
        "effort"_a = 7);

  m.def("jxl_to_jpeg",
        &jxl_to_jpeg,
        "Reconstruct original JPEG data from JXL bytes (if recompressed).",
        "data"_a);

  m.def("jpeg_to_jxl_file",
        &jpeg_to_jxl_file,
        "Direct C++ file-to-file lossless JPEG to JXL recompression (zero Python heap overhead).",
        "in_path"_a,
        "out_path"_a,
        "effort"_a = 7);

  m.def("jxl_to_jpeg_file",
        &jxl_to_jpeg_file,
        "Direct C++ file-to-file lossless JXL to JPEG reconstruction (zero Python heap overhead).",
        "in_path"_a,
        "out_path"_a);
}
