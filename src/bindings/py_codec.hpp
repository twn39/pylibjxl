#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <optional>
#include <stdexcept>
#include <string>

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
                      int threads = 0,
                      int pool_size = 0,
                      std::optional<double> timeout = std::nullopt,
                      double idle_timeout = 30.0)
      : effort_(std::clamp(effort, 1, 11)),
        distance_(lossless ? 0.0F : std::clamp(distance, 0.0F, 25.0F)), lossless_(lossless),
        decoding_speed_(std::clamp(decoding_speed, 0, 4)),
        default_timeout_(timeout.has_value()
                             ? std::make_optional(std::chrono::milliseconds(
                                   static_cast<int64_t>(*timeout * 1000.0)))
                             : std::nullopt) {
    size_t tpr = threads > 0 ? static_cast<size_t>(threads) : 0;
    size_t ps = pool_size > 0 ? static_cast<size_t>(pool_size) : 0;
    auto it = std::chrono::milliseconds(static_cast<int64_t>(std::max(0.0, idle_timeout) * 1000.0));
    pool_ = std::make_unique<RunnerPool>(ps, tpr, it);
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
                         nb::handle icc,
                         std::optional<double> timeout = std::nullopt) {
    check_closed();
    int eff = effort.value_or(effort_);
    bool ll = lossless.value_or(lossless_);
    float dist = distance.value_or(ll ? 0.0F : distance_);
    int ds = decoding_speed.value_or(decoding_speed_);
    return encode_impl(
        input, eff, dist, ll, ds, exif, xmp, jumbf, icc, *pool_, resolve_timeout(timeout));
  }

  nb::object decode_image(nb::handle data,
                          bool metadata,
                          std::optional<nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu>> out,
                          std::optional<double> timeout = std::nullopt) {
    check_closed();
    return decode_impl(data, metadata, out, *pool_, resolve_timeout(timeout));
  }

  nb::dict probe_image(nb::handle data) {
    check_closed();
    return probe(data);
  }

  nb::bytes encode_jpeg_image(nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu> input,
                              int quality) {
    check_closed();
    return encode_jpeg(input, quality);
  }

  nb::object
  decode_jpeg_image(nb::handle data,
                    std::optional<nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu>> out) {
    check_closed();
    return decode_jpeg(data, out);
  }

  nb::bytes jpeg_to_jxl_image(nb::handle jpeg_data,
                              std::optional<int> effort,
                              std::optional<double> timeout = std::nullopt) {
    check_closed();
    return jpeg_to_jxl_impl(
        jpeg_data, effort.value_or(effort_), *pool_, resolve_timeout(timeout));
  }

  nb::bytes jxl_to_jpeg_image(nb::handle jxl_data,
                              std::optional<double> timeout = std::nullopt) {
    check_closed();
    return jxl_to_jpeg_impl(jxl_data, *pool_, resolve_timeout(timeout));
  }

  void jpeg_to_jxl_file_image(const std::string &in_path,
                              const std::string &out_path,
                              std::optional<int> effort,
                              std::optional<double> timeout = std::nullopt) {
    check_closed();
    jpeg_to_jxl_file_impl(
        in_path, out_path, effort.value_or(effort_), *pool_, resolve_timeout(timeout));
  }

  void jxl_to_jpeg_file_image(const std::string &in_path,
                              const std::string &out_path,
                              std::optional<double> timeout = std::nullopt) {
    check_closed();
    jxl_to_jpeg_file_impl(in_path, out_path, *pool_, resolve_timeout(timeout));
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

  [[nodiscard]] size_t pool_size() const {
    check_closed();
    return pool_->pool_size();
  }

  [[nodiscard]] size_t total_runners() const {
    check_closed();
    return pool_->total_runners();
  }

  [[nodiscard]] size_t available_runners() const {
    check_closed();
    return pool_->available_runners();
  }

  [[nodiscard]] size_t in_use_runners() const {
    check_closed();
    return pool_->in_use_runners();
  }

  [[nodiscard]] size_t threads_per_runner() const {
    check_closed();
    return pool_->threads_per_runner();
  }

private:
  void check_closed() const {
    if (closed_) {
      throw std::runtime_error("Cannot use a closed JXL codec");
    }
  }

  [[nodiscard]] std::optional<std::chrono::milliseconds>
  resolve_timeout(std::optional<double> timeout_sec) const {
    if (timeout_sec.has_value()) {
      return std::chrono::milliseconds(static_cast<int64_t>(*timeout_sec * 1000.0));
    }
    return default_timeout_;
  }

  int effort_;
  float distance_;
  bool lossless_;
  int decoding_speed_;
  std::optional<std::chrono::milliseconds> default_timeout_;
  bool closed_ = false;
  std::unique_ptr<RunnerPool> pool_;
};

} // namespace pylibjxl
