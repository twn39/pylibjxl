#pragma once

#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace pylibjxl {

nanobind::bytes
encode_jpeg(nanobind::ndarray<uint8_t, nanobind::c_contig, nanobind::device::cpu> input,
            int quality = 95);

nanobind::ndarray<uint8_t, nanobind::numpy, nanobind::device::cpu>
decode_jpeg(nanobind::bytes data);

} // namespace pylibjxl
