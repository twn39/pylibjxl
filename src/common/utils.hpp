#pragma once

#include <cstdint>
#include <nanobind/nanobind.h>
#include <vector>

namespace pylibjxl {

inline std::vector<uint8_t> extract_optional_bytes(const nanobind::handle &obj) {
  if (obj.is_none()) {
    return {};
  }
  const auto *ptr = reinterpret_cast<const uint8_t *>(PyBytes_AsString(obj.ptr()));
  const auto size = static_cast<size_t>(PyBytes_Size(obj.ptr()));
  return {ptr, ptr + size};
}

} // namespace pylibjxl
