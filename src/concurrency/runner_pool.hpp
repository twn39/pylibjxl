#pragma once

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "common/deleters.hpp"

namespace pylibjxl {

size_t suggest_threads(uint64_t xsize, uint64_t ysize);

// A pool of JxlResizableParallelRunner instances for thread-safe concurrent operations.
// Each runner is independently usable, allowing true parallel JXL encode/decode.
class RunnerPool {
public:
  // pool_size: number of runners in the pool (= max concurrent operations)
  // threads_per_runner: threads each runner uses internally
  // When pool_size=0 and threads_per_runner=0: auto-balance based on CPU cores
  // When pool_size=0 and threads_per_runner>0: pool_size = max(1, cores / threads_per_runner)
  explicit RunnerPool(size_t pool_size = 0, size_t threads_per_runner = 0);

  RunnerPool(const RunnerPool &) = delete;
  RunnerPool &operator=(const RunnerPool &) = delete;
  RunnerPool(RunnerPool &&) = delete;
  RunnerPool &operator=(RunnerPool &&) = delete;
  ~RunnerPool() = default;

  // Acquire a runner from the pool. Blocks if none are available.
  JxlRunnerPtr acquire();

  // Release a runner back to the pool.
  void release(JxlRunnerPtr runner);

  // Destroy all pooled runners (for close()).
  void clear();

private:
  std::vector<JxlRunnerPtr> pool_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

// RAII guard: automatically releases the runner back to the pool on destruction.
class RunnerGuard {
public:
  explicit RunnerGuard(RunnerPool &pool);
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
