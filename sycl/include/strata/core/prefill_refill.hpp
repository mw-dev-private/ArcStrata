#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/expert_cache.hpp"
#include "strata/core/gguf_expert_source.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

namespace strata::core {

inline bool prefill_refill_async_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_PREFILL_REFILL_ASYNC");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

// Only the loan-refill callers use this helper. Owned pinned sources remain valid until
// the batch drains, independent of cache hits and wraparound in the GGUF source's ring.
class PrefillRefillBatch {
public:
    explicit PrefillRefillBatch(size_t stride) : stride_(stride), queue_(&dpct::get_in_order_queue()) {
        const char* value = std::getenv("STRATA_PREFILL_REFILL_VERIFY");
        verify_ = value && std::strcmp(value, "1") == 0;
    }
    PrefillRefillBatch(const PrefillRefillBatch&) = delete;
    PrefillRefillBatch& operator=(const PrefillRefillBatch&) = delete;
    ~PrefillRefillBatch() noexcept {
        if (!staging_ && !readback_) return;
        try {
            queue_->wait_and_throw();
            if (readback_) sycl::free(readback_, *queue_);
            if (staging_) sycl::free(staging_, *queue_);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "prefill async refill teardown: %s\n", error.what());
            std::abort(); // Never release a source whose DMA completion is unknown.
        }
    }

    bool push(ExpertCache& cache, int32_t slot, ExpertSource& source, int64_t layer, int64_t expert,
              size_t bytes, std::string& err) {
        if (bytes == 0 || bytes > stride_ || slot < 0 || slot >= cache.slots() ||
            bytes > cache.slot_offset(slot + 1) - cache.slot_offset(slot)) {
            err = "prefill async refill: invalid source, slot or byte count";
            return false;
        }
        if (!staging_ && !allocate(err)) return false;
        if (used_ == capacity_ && !drain(err)) return false;
        uint8_t* owned = staging_ + used_ * stride_;
        auto* gguf = dynamic_cast<GgufExpertSource*>(&source);
        if (gguf && !source.pinned(layer, expert)) {
            // Read the three GGUF slices straight into owned DMA storage, avoiding a second host copy.
            if (!gguf->read_into(layer, expert, owned, bytes)) {
                err = "prefill async refill: GGUF read_into failed";
                return false;
            }
        } else {
            const uint8_t* blob = source.blob(layer, expert);
            if (!blob) { err = "prefill async refill: source blob is null"; return false; }
            std::memcpy(owned, blob, bytes);
        }
        if (!cache.fill_slot(slot, owned, queue_, err, (int64_t) bytes)) return false;
        pending_[used_++] = {cache.device_slot(slot), bytes, slot};
        ++copies_;
        bytes_ += bytes;
        return true;
    }

    bool finish(std::string& err) {
        if (!drain(err)) return false;
        if (copies_ > 0)
            std::fprintf(stderr, "strata: prefill async refill: %zu slots, %zu bytes, %zu batches, "
                         "host staging %.2f MiB, extra device 0 MiB, verify %s (%zu slots)\n",
                         copies_, bytes_, batches_, (double) (capacity_ * stride_) / 1048576.0,
                         verify_ ? "BITWISE EQUAL" : "off", verified_);
        return true;
    }

private:
    bool allocate(std::string& err) {
        constexpr size_t host_limit = 64 * 1024 * 1024;
        if (stride_ == 0 || stride_ > host_limit) {
            err = "prefill async refill: invalid staging stride";
            return false;
        }
        capacity_ = std::min<size_t>(16, host_limit / stride_);
        try {
            staging_ = sycl::malloc_host<uint8_t>(capacity_ * stride_, *queue_);
            if (verify_) readback_ = sycl::malloc_host<uint8_t>(stride_, *queue_);
        } catch (const std::exception& error) {
            err = std::string("prefill async refill allocation: ") + error.what();
            return false;
        }
        if (!staging_ || (verify_ && !readback_)) {
            err = "prefill async refill: pinned staging allocation failed";
            return false;
        }
        return true;
    }

    bool drain(std::string& err) {
        if (used_ == 0) return true;
        try {
            queue_->wait_and_throw();
            if (verify_) for (size_t i = 0; i < used_; ++i) {
                const Pending& item = pending_[i];
                queue_->memcpy(readback_, item.destination, item.bytes).wait_and_throw();
                if (std::memcmp(readback_, staging_ + i * stride_, item.bytes) != 0) {
                    err = "prefill async refill: byte mismatch at slot " + std::to_string(item.slot);
                    return false;
                }
                ++verified_;
            }
        } catch (const std::exception& error) {
            err = std::string("prefill async refill completion: ") + error.what();
            return false;
        }
        used_ = 0;
        ++batches_;
        return true;
    }

    struct Pending { uint8_t* destination; size_t bytes; int32_t slot; } pending_[16]{};
    size_t stride_, capacity_ = 0, used_ = 0, copies_ = 0, bytes_ = 0, batches_ = 0, verified_ = 0;
    dpct::queue_ptr queue_;
    uint8_t* staging_ = nullptr;
    uint8_t* readback_ = nullptr;
    bool verify_ = false;
};
} // namespace strata::core
