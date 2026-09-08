#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <optional>
#include <stdexcept>

#include "codecs/jpeg_ops.hpp"
#include "codecs/jxl_ops.hpp"
#include "codecs/transcode.hpp"
#include "concurrency/runner_pool.hpp"

namespace nb = nanobind;

namespace pylibjxl {

class PyJxlCodec {
public:
  explicit PyJxlCodec(int effort = 7,
                      float distance = 1.0F,
                      bool lossless = false,
                      int decoding_speed = 0,
                      int threads = 0)
      : effort_(std::clamp(effort, 1, 11)),
        distance_(lossless ? 0.0F : std::clamp(distance, 0.0F, 25.0F)), lossless_(lossless),
        decoding_speed_(std::clamp(decoding_speed, 0, 4)) {
    // threads param controls threads_per_runner; pool_size is auto-calculated.
    // threads=0 → auto-balance (default).
    size_t tpr = threads > 0 ? static_cast<size_t>(threads) : 0;
    pool_ = std::make_unique<RunnerPool>(0, tpr);
  }

  ~PyJxlCodec() { close(); }

  PyJxlCodec(const PyJxlCodec &) = delete;
  PyJxlCodec &operator=(const PyJxlCodec &) = delete;
  PyJxlCodec(PyJxlCodec &&) = delete;
  PyJxlCodec &operator=(PyJxlCodec &&) = delete;

  nb::bytes encode_image(nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu> input,
                         std::optional<int> effort,
                         std::optional<float> distance,
                         std::optional<bool> lossless,
                         std::optional<int> decoding_speed,
                         nb::handle exif,
                         nb::handle xmp,
                         nb::handle jumbf,
                         nb::handle icc) {
    check_closed();
    int eff = effort.value_or(effort_);
    bool ll = lossless.value_or(lossless_);
    float dist = distance.value_or(ll ? 0.0F : distance_);
    int ds = decoding_speed.value_or(decoding_speed_);
    return encode_impl(input, eff, dist, ll, ds, exif, xmp, jumbf, icc, *pool_);
  }

  nb::object decode_image(nb::bytes data, bool metadata) {
    check_closed();
    return decode_impl(data, metadata, *pool_);
  }

  nb::bytes encode_jpeg_image(nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu> input,
                              int quality) {
    check_closed();
    return encode_jpeg(input, quality);
  }

  nb::ndarray<uint8_t, nb::numpy, nb::device::cpu> decode_jpeg_image(nb::bytes data) {
    check_closed();
    return decode_jpeg(data);
  }

  nb::bytes jpeg_to_jxl_image(nb::bytes jpeg_data, std::optional<int> effort) {
    check_closed();
    return jpeg_to_jxl_impl(jpeg_data, effort.value_or(effort_), *pool_);
  }

  nb::bytes jxl_to_jpeg_image(nb::bytes jxl_data) {
    check_closed();
    return jxl_to_jpeg_impl(jxl_data, *pool_);
  }

  PyJxlCodec &enter() {
    check_closed();
    return *this;
  }

  void exit(nb::handle /*exc_type*/, nb::handle /*exc_val*/, nb::handle /*exc_tb*/) { close(); }

  void close() {
    closed_ = true;
    if (pool_) {
      pool_->clear();
    }
  }

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
  int decoding_speed_;
  bool closed_ = false;
  std::unique_ptr<RunnerPool> pool_;
};

} // namespace pylibjxl
