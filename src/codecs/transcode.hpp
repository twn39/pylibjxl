#pragma once

#include <chrono>
#include <nanobind/nanobind.h>
#include <optional>
#include <string>

#include "concurrency/runner_pool.hpp"

namespace pylibjxl {

nanobind::bytes jpeg_to_jxl_impl(nanobind::handle jpeg_data,
                                 int effort,
                                 RunnerPool &pool,
                                 std::optional<std::chrono::milliseconds> timeout = std::nullopt);
nanobind::bytes jpeg_to_jxl(nanobind::handle jpeg_data,
                            int effort = 7,
                            std::optional<std::chrono::milliseconds> timeout = std::nullopt);

nanobind::bytes jxl_to_jpeg_impl(nanobind::handle jxl_data,
                                 RunnerPool &pool,
                                 std::optional<std::chrono::milliseconds> timeout = std::nullopt);
nanobind::bytes jxl_to_jpeg(nanobind::handle jxl_data,
                            std::optional<std::chrono::milliseconds> timeout = std::nullopt);

void jpeg_to_jxl_file_impl(const std::string &in_path,
                           const std::string &out_path,
                           int effort,
                           RunnerPool &pool,
                           std::optional<std::chrono::milliseconds> timeout = std::nullopt);
void jpeg_to_jxl_file(const std::string &in_path,
                      const std::string &out_path,
                      int effort = 7,
                      std::optional<std::chrono::milliseconds> timeout = std::nullopt);

void jxl_to_jpeg_file_impl(const std::string &in_path,
                           const std::string &out_path,
                           RunnerPool &pool,
                           std::optional<std::chrono::milliseconds> timeout = std::nullopt);
void jxl_to_jpeg_file(const std::string &in_path,
                      const std::string &out_path,
                      std::optional<std::chrono::milliseconds> timeout = std::nullopt);

} // namespace pylibjxl
