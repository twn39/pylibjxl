#pragma once

#include <nanobind/nanobind.h>
#include <string>

#include "concurrency/runner_pool.hpp"

namespace pylibjxl {

nanobind::bytes jpeg_to_jxl_impl(nanobind::handle jpeg_data, int effort, RunnerPool &pool);
nanobind::bytes jpeg_to_jxl(nanobind::handle jpeg_data, int effort = 7);

nanobind::bytes jxl_to_jpeg_impl(nanobind::handle jxl_data, RunnerPool &pool);
nanobind::bytes jxl_to_jpeg(nanobind::handle jxl_data);

void jpeg_to_jxl_file_impl(const std::string &in_path,
                           const std::string &out_path,
                           int effort,
                           RunnerPool &pool);
void jpeg_to_jxl_file(const std::string &in_path, const std::string &out_path, int effort = 7);

void jxl_to_jpeg_file_impl(const std::string &in_path,
                           const std::string &out_path,
                           RunnerPool &pool);
void jxl_to_jpeg_file(const std::string &in_path, const std::string &out_path);

} // namespace pylibjxl
