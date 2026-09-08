#pragma once

#include <nanobind/nanobind.h>

#include "concurrency/runner_pool.hpp"

namespace pylibjxl {

nanobind::bytes jpeg_to_jxl_impl(nanobind::bytes jpeg_data, int effort, RunnerPool &pool);
nanobind::bytes jpeg_to_jxl(nanobind::bytes jpeg_data, int effort = 7);

nanobind::bytes jxl_to_jpeg_impl(nanobind::bytes jxl_data, RunnerPool &pool);
nanobind::bytes jxl_to_jpeg(nanobind::bytes jxl_data);

} // namespace pylibjxl
