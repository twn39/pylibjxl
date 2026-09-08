#pragma once

#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <optional>

namespace pylibjxl {

nanobind::bytes
encode_jpeg(nanobind::ndarray<uint8_t, nanobind::c_contig, nanobind::device::cpu> input,
            int quality = 95);

nanobind::object decode_jpeg(
    nanobind::handle data,
    std::optional<nanobind::ndarray<uint8_t, nanobind::c_contig, nanobind::device::cpu>> out =
        std::nullopt);

} // namespace pylibjxl
