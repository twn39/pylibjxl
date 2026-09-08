#pragma once

#include <chrono>
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "concurrency/runner_pool.hpp"

namespace pylibjxl {

nanobind::bytes
encode_impl(nanobind::ndarray<uint8_t, nanobind::c_contig, nanobind::device::cpu> input,
            int effort,
            float distance,
            bool lossless,
            int decoding_speed,
            nanobind::handle exif,
            nanobind::handle xmp,
            nanobind::handle jumbf,
            nanobind::handle icc,
            RunnerPool &pool,
            std::optional<std::chrono::milliseconds> timeout = std::nullopt);

nanobind::bytes encode(nanobind::ndarray<uint8_t, nanobind::c_contig, nanobind::device::cpu> input,
                       int effort = 7,
                       float distance = 1.0F,
                       bool lossless = false,
                       int decoding_speed = 0,
                       nanobind::handle exif = nanobind::none(),
                       nanobind::handle xmp = nanobind::none(),
                       nanobind::handle jumbf = nanobind::none(),
                       nanobind::handle icc = nanobind::none(),
                       std::optional<std::chrono::milliseconds> timeout = std::nullopt);

nanobind::object decode_impl(
    nanobind::handle data,
    bool metadata,
    std::optional<nanobind::ndarray<uint8_t, nanobind::c_contig, nanobind::device::cpu>> out,
    RunnerPool &pool,
    std::optional<std::chrono::milliseconds> timeout = std::nullopt);

nanobind::object
decode(nanobind::handle data,
       bool metadata = false,
       std::optional<nanobind::ndarray<uint8_t, nanobind::c_contig, nanobind::device::cpu>> out =
           std::nullopt,
       std::optional<std::chrono::milliseconds> timeout = std::nullopt);

} // namespace pylibjxl
