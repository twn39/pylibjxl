#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pylibjxl {

// Thread-local scratch buffer with high-watermark reclamation to avoid memory bloat.
// Default high watermark: 32 MB.
inline constexpr size_t kDefaultBufferWatermark = 32 * 1024 * 1024;

class ThreadLocalBuffer {
public:
    static std::vector<uint8_t>& acquire(size_t watermark = kDefaultBufferWatermark) {
        thread_local std::vector<uint8_t> buffer;
        if (buffer.capacity() > watermark) {
            std::vector<uint8_t>().swap(buffer); // Release excess capacity back to OS
        }
        buffer.clear();
        return buffer;
    }
};

} // namespace pylibjxl
