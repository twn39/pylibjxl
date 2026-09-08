#include "concurrency/runner_pool.hpp"

#include <algorithm>
#include <stdexcept>
#include <thread>

namespace pylibjxl {

size_t suggest_threads(uint64_t xsize, uint64_t ysize) {
  size_t threads = JxlResizableParallelRunnerSuggestThreads(xsize, ysize);
  if (threads == 0) {
    threads = std::max<size_t>(1, std::thread::hardware_concurrency());
  }
  return threads;
}

RunnerPool::RunnerPool(size_t pool_size, size_t threads_per_runner) {
  size_t cores = std::max<size_t>(1, std::thread::hardware_concurrency());
  if (threads_per_runner == 0 && pool_size == 0) {
    // Auto-balance: each runner gets enough threads for good single-task speed,
    // while allowing multiple concurrent operations.
    threads_per_runner = std::max<size_t>(2, cores / 4);
    pool_size = std::max<size_t>(1, cores / threads_per_runner);
  } else if (pool_size == 0) {
    pool_size = std::max<size_t>(1, cores / threads_per_runner);
  } else if (threads_per_runner == 0) {
    threads_per_runner = std::max<size_t>(1, cores / pool_size);
  }
  for (size_t i = 0; i < pool_size; ++i) {
    JxlRunnerPtr runner(JxlResizableParallelRunnerCreate(nullptr));
    if (runner == nullptr) {
      throw std::runtime_error("JxlResizableParallelRunnerCreate failed");
    }
    JxlResizableParallelRunnerSetThreads(runner.get(), threads_per_runner);
    pool_.push_back(std::move(runner));
  }
}

JxlRunnerPtr RunnerPool::acquire() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return !pool_.empty(); });
  JxlRunnerPtr runner = std::move(pool_.back());
  pool_.pop_back();
  return runner;
}

void RunnerPool::release(JxlRunnerPtr runner) {
  if (runner == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  pool_.emplace_back(std::move(runner));
  cv_.notify_one();
}

void RunnerPool::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  pool_.clear();
}

RunnerGuard::RunnerGuard(RunnerPool &pool) : pool_(pool), runner_(pool.acquire()) {}

RunnerGuard::~RunnerGuard() {
  pool_.release(std::move(runner_));
}

void *RunnerGuard::get() const {
  return runner_.get();
}

namespace {
// Global runner pool for free functions (lazily initialized).
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::unique_ptr<RunnerPool> g_pool;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::once_flag g_pool_init_flag;
} // namespace

RunnerPool &global_pool() {
  std::call_once(g_pool_init_flag, [] {
    // Auto-balanced: uses default (0, 0) which splits cores between pool_size and tpr
    g_pool = std::make_unique<RunnerPool>();
  });
  return *g_pool;
}

} // namespace pylibjxl
