#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/deleters.hpp"

namespace pylibjxl {

// Exception thrown when acquiring a runner from RunnerPool exceeds the timeout limit.
class CodecTimeoutError : public std::runtime_error {
public:
  explicit CodecTimeoutError(const std::string &msg) : std::runtime_error(msg) {}
  explicit CodecTimeoutError(const char *msg) : std::runtime_error(msg) {}
};

size_t suggest_threads(uint64_t xsize, uint64_t ysize);

// An elastic, thread-safe pool of JxlResizableParallelRunner instances.
// Supports on-demand lazy allocation, idle reaping, bounded concurrency, and acquire timeouts.
class RunnerPool {
public:
  // pool_size: maximum number of runners in the pool (= max concurrent operations)
  // threads_per_runner: internal worker threads each runner uses
  // idle_timeout: duration after which idle runners beyond min_pool_size are reaped
  explicit RunnerPool(size_t pool_size = 0,
                      size_t threads_per_runner = 0,
                      std::chrono::milliseconds idle_timeout = std::chrono::milliseconds(30000));

  RunnerPool(const RunnerPool &) = delete;
  RunnerPool &operator=(const RunnerPool &) = delete;
  RunnerPool(RunnerPool &&) = delete;
  RunnerPool &operator=(RunnerPool &&) = delete;
  ~RunnerPool() = default;

  // Acquire a runner from the pool.
  // If timeout is specified, waits up to timeout before throwing CodecTimeoutError.
  JxlRunnerPtr acquire(std::optional<std::chrono::milliseconds> timeout = std::nullopt);

  // Release a runner back to the pool.
  void release(JxlRunnerPtr runner);

  // Destroy all pooled runners and reset allocation count (for close()).
  void clear();

  // Introspection & telemetry metrics
  [[nodiscard]] size_t pool_size() const;
  [[nodiscard]] size_t total_runners() const;
  [[nodiscard]] size_t available_runners() const;
  [[nodiscard]] size_t in_use_runners() const;
  [[nodiscard]] size_t threads_per_runner() const;

private:
  struct PooledRunner {
    JxlRunnerPtr runner;
    std::chrono::steady_clock::time_point last_used;
  };

  void reap_idle_runners_locked(std::chrono::steady_clock::time_point now);

  size_t max_pool_size_ = 0;
  size_t threads_per_runner_ = 0;
  size_t total_created_ = 0;
  std::chrono::milliseconds idle_timeout_;
  std::vector<PooledRunner> pool_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
};

// RAII guard: automatically releases the runner back to the pool on destruction.
class RunnerGuard {
public:
  explicit RunnerGuard(RunnerPool &pool,
                       std::optional<std::chrono::milliseconds> timeout = std::nullopt);
  ~RunnerGuard();

  RunnerGuard(const RunnerGuard &) = delete;
  RunnerGuard &operator=(const RunnerGuard &) = delete;
  RunnerGuard(RunnerGuard &&) = delete;
  RunnerGuard &operator=(RunnerGuard &&) = delete;

  [[nodiscard]] void *get() const;

private:
  RunnerPool &pool_;
  JxlRunnerPtr runner_;
};

// Global runner pool for free functions (lazily initialized).
RunnerPool &global_pool();

} // namespace pylibjxl
