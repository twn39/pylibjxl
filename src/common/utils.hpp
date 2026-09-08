#pragma once

#include <cstdint>
#include <nanobind/nanobind.h>
#include <stdexcept>
#include <vector>

namespace pylibjxl {

// RAII wrapper for Python Buffer Protocol (Py_buffer)
struct ScopedPyBuffer {
  Py_buffer view{};
  bool valid = false;

  explicit ScopedPyBuffer(const nanobind::handle &obj) {
    if (obj.is_none()) {
      return;
    }
    if (PyObject_GetBuffer(obj.ptr(), &view, PyBUF_SIMPLE) == 0) {
      valid = true;
    } else {
      PyErr_Clear();
      throw std::invalid_argument("Object does not support Python buffer protocol or cannot be "
                                  "read as a contiguous byte buffer");
    }
  }

  ~ScopedPyBuffer() {
    if (valid) {
      PyBuffer_Release(&view);
    }
  }

  ScopedPyBuffer(const ScopedPyBuffer &) = delete;
  ScopedPyBuffer &operator=(const ScopedPyBuffer &) = delete;
  ScopedPyBuffer(ScopedPyBuffer &&) = delete;
  ScopedPyBuffer &operator=(ScopedPyBuffer &&) = delete;

  [[nodiscard]] const uint8_t *data() const { return reinterpret_cast<const uint8_t *>(view.buf); }

  [[nodiscard]] size_t size() const { return static_cast<size_t>(view.len); }
};

inline std::vector<uint8_t> extract_optional_bytes(const nanobind::handle &obj) {
  if (obj.is_none()) {
    return {};
  }
  ScopedPyBuffer buf(obj);
  return {buf.data(), buf.data() + buf.size()};
}

} // namespace pylibjxl
