#include "concurrency/runner_pool.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>

namespace pylibjxl {

size_t suggest_threads(uint64_t xsize, uint64_t ysize) {
  size_t threads = JxlResizableParallelRunnerSuggestThreads(xsize, ysize);
  if (threads == 0) {
    threads = std::max<size_t>(1, std::thread::hardware_concurrency());
  }
  return threads;
}

RunnerPool::RunnerPool(size_t pool_size,
                       size_t threads_per_runner,
                       std::chrono::milliseconds idle_timeout)
    : idle_timeout_(idle_timeout) {
  size_t cores = std::max<size_t>(1, std::thread::hardware_concurrency());
  if (threads_per_runner == 0 && pool_size == 0) {
    // Auto-balance: max_pool_size * threads_per_runner <= cores
    threads_per_runner_ = std::max<size_t>(2, cores / 4);
    max_pool_size_ = std::max<size_t>(1, cores / threads_per_runner_);
  } else if (pool_size == 0) {
    threads_per_runner_ = threads_per_runner;
    max_pool_size_ = std::max<size_t>(1, cores / threads_per_runner_);
  } else if (threads_per_runner == 0) {
    max_pool_size_ = pool_size;
    threads_per_runner_ = std::max<size_t>(1, cores / max_pool_size_);
  } else {
    max_pool_size_ = pool_size;
    threads_per_runner_ = threads_per_runner;
  }

  // Pre-allocate 1 runner eagerly so single-operation tasks have zero warm-up latency
  JxlRunnerPtr runner(JxlResizableParallelRunnerCreate(nullptr));
  if (runner != nullptr) {
    JxlResizableParallelRunnerSetThreads(runner.get(), threads_per_runner_);
    pool_.push_back({std::move(runner), std::chrono::steady_clock::now()});
    total_created_ = 1;
  }
}

void RunnerPool::reap_idle_runners_locked(std::chrono::steady_clock::time_point now) {
  if (pool_.size() <= 1 || idle_timeout_.count() <= 0) {
    return;
  }
  // Keep at least 1 warm runner, reap idle runners beyond index 0 that exceed idle_timeout_
  auto it = std::remove_if(pool_.begin() + 1, pool_.end(), [&](const PooledRunner &item) {
    if (now - item.last_used > idle_timeout_) {
      --total_created_;
      return true;
    }
    return false;
  });
  pool_.erase(it, pool_.end());
}

JxlRunnerPtr RunnerPool::acquire(std::optional<std::chrono::milliseconds> timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto now = std::chrono::steady_clock::now();
  reap_idle_runners_locked(now);

  if (!pool_.empty()) {
    JxlRunnerPtr runner = std::move(pool_.back().runner);
    pool_.pop_back();
    return runner;
  }

  // Lazily scale up if below max_pool_size_
  if (total_created_ < max_pool_size_) {
    JxlRunnerPtr runner(JxlResizableParallelRunnerCreate(nullptr));
    if (runner == nullptr) {
      throw std::runtime_error("JxlResizableParallelRunnerCreate failed");
    }
    JxlResizableParallelRunnerSetThreads(runner.get(), threads_per_runner_);
    ++total_created_;
    return runner;
  }

  // Pool capacity reached: wait for a runner to be released
  if (timeout.has_value()) {
    bool acquired = cv_.wait_for(lock, *timeout, [this] { return !pool_.empty(); });
    if (!acquired) {
      throw CodecTimeoutError("RunnerPool acquisition timed out after " +
                              std::to_string(timeout->count()) + "ms");
    }
  } else {
    cv_.wait(lock, [this] { return !pool_.empty(); });
  }

  JxlRunnerPtr runner = std::move(pool_.back().runner);
  pool_.pop_back();
  return runner;
}

void RunnerPool::release(JxlRunnerPtr runner) {
  if (runner == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  pool_.push_back({std::move(runner), std::chrono::steady_clock::now()});
  cv_.notify_one();
}

void RunnerPool::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  pool_.clear();
  total_created_ = 0;
}

size_t RunnerPool::pool_size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return max_pool_size_;
}

size_t RunnerPool::total_runners() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return total_created_;
}

size_t RunnerPool::available_runners() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return pool_.size();
}

size_t RunnerPool::in_use_runners() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return total_created_ >= pool_.size() ? (total_created_ - pool_.size()) : 0;
}

size_t RunnerPool::threads_per_runner() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return threads_per_runner_;
}

RunnerGuard::RunnerGuard(RunnerPool &pool, std::optional<std::chrono::milliseconds> timeout)
    : pool_(pool), runner_(pool.acquire(timeout)) {}

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
