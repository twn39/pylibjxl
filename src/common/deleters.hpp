#pragma once

#include <memory>
#include <turbojpeg.h>

#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/resizable_parallel_runner.h>

namespace pylibjxl {

struct JxlEncoderDeleter {
  void operator()(JxlEncoder *p) const { JxlEncoderDestroy(p); }
};
using JxlEncoderPtr = std::unique_ptr<JxlEncoder, JxlEncoderDeleter>;

struct JxlDecoderDeleter {
  void operator()(JxlDecoder *p) const { JxlDecoderDestroy(p); }
};
using JxlDecoderPtr = std::unique_ptr<JxlDecoder, JxlDecoderDeleter>;

struct JxlRunnerDeleter {
  void operator()(void *p) const { JxlResizableParallelRunnerDestroy(p); }
};
using JxlRunnerPtr = std::unique_ptr<void, JxlRunnerDeleter>;

struct TjDeleter {
  void operator()(void *p) const { tjDestroy(p); }
};
using TjPtr = std::unique_ptr<void, TjDeleter>;

struct TjFree {
  void operator()(const unsigned char *p) const {
    tjFree(const_cast<unsigned char *>(p)); // NOLINT(cppcoreguidelines-pro-type-const-cast)
  }
};
using TjBufPtr = std::unique_ptr<unsigned char, TjFree>;

} // namespace pylibjxl
