// ecap_parity - the verify window's expert-cap redirect against its contract (SYCL port of upstream PR #3545).
//
// `verify_ecap` (STRATA_VERIFY_ECAP) rewrites the window's routing in place before resident_plan: token 0 keeps all
// k picks, the speculative tail keeps the top (cap - k) entries by (weight desc, index asc), and every dropped entry
// is redirected round-robin onto a kept expert with weight 0.  Speculative decoding's exactness for the committed
// position and the plan's residency requirement both hang on the kernel keeping its promises:
//   * cap >= the entry count (or n <= k, or cap <= k): a no-op - ids and weights bitwise unchanged
//   * token 0's entries are never touched
//   * every kept entry keeps its id and weight; every dropped entry's weight is exactly 0 and its id is one of the
//     kept set's (the plan never meets an expert the uncapped window would not have named)
//   * the same input gives the same output (the fold is deterministic)
#include "strata/kernels/verify_kernels.hpp"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    sycl::queue& q = dpct::get_in_order_queue();   // the queue verify_ecap uses (q_of(nullptr))
    std::printf("ecap_parity: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int E = 256, K = 10;
    int bad = 0;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> w_dist(0.0f, 1.0f);
    for (int T : {2, 3, 4, 6}) {
        const int n = T * K;
        std::vector<int32_t> ids(n), ids0;
        std::vector<float> w(n), w0;
        for (int t = 0; t < T; ++t) {              // distinct ids per token, like the router's top-k
            std::vector<int> pool(E);
            for (int i = 0; i < E; ++i) pool[i] = i;
            std::shuffle(pool.begin(), pool.end(), rng);
            std::sort(pool.begin(), pool.begin() + K);
            for (int j = 0; j < K; ++j) ids[t * K + j] = pool[j];
        }
        for (auto& v : w) v = w_dist(rng);
        ids0 = ids; w0 = w;

        int32_t* d_ids = sycl::malloc_device<int32_t>(n, q);
        float* d_w = sycl::malloc_device<float>(n, q);
        auto up = [&] {
            q.memcpy(d_ids, ids.data(), n * sizeof(int32_t)).wait();
            q.memcpy(d_w, w.data(), n * sizeof(float)).wait();
        };
        auto down = [&] {
            q.memcpy(ids.data(), d_ids, n * sizeof(int32_t)).wait();
            q.memcpy(w.data(), d_w, n * sizeof(float)).wait();
        };
        auto same = [&] {
            return std::memcmp(ids.data(), ids0.data(), n * sizeof(int32_t)) == 0 &&
                   std::memcmp(w.data(), w0.data(), n * sizeof(float)) == 0;
        };

        up();                                       // a cap that keeps everything is a no-op, bitwise
        strata::kernels::verify_ecap(d_ids, d_w, n, K, n + 7, nullptr);
        q.wait(); down();
        std::printf("  cap>=n      T %d  %s\n", T, same() ? "no-op" : "*** CHANGED ***");
        if (!same()) ++bad;

        up();                                       // n <= k or cap <= k: a no-op
        strata::kernels::verify_ecap(d_ids, d_w, n, K, K, nullptr);
        strata::kernels::verify_ecap(d_ids, d_w, K, K, 20, nullptr);
        q.wait(); down();
        std::printf("  no-cap      T %d  %s\n", T, same() ? "no-op" : "*** CHANGED ***");
        if (!same()) ++bad;

        for (int cap : {K + 1, 16, 20, 26}) {
            if (cap >= n) continue;
            std::vector<int> tail(n - K);           // the expected kept set: token 0's k, then top (cap-k) by w desc
            for (int j = K; j < n; ++j) tail[j - K] = j;
            std::stable_sort(tail.begin(), tail.end(), [&](int a, int b) { return w0[a] > w0[b]; });
            std::vector<char> kept(n, 0);
            for (int j = 0; j < K; ++j) kept[j] = 1;
            for (int i = 0; i < cap - K && i < (int) tail.size(); ++i) kept[tail[i]] = 1;

            ids = ids0; w = w0;
            up(); strata::kernels::verify_ecap(d_ids, d_w, n, K, cap, nullptr); q.wait(); down();
            auto d1 = ids; auto ww1 = w;
            ids = ids0; w = w0;
            up(); strata::kernels::verify_ecap(d_ids, d_w, n, K, cap, nullptr); q.wait(); down();
            int errs = 0;
            for (int i = 0; i < n; ++i) {
                const bool is_kept = kept[i] != 0;
                if (i < K && (ids[i] != ids0[i] || w[i] != w0[i])) ++errs;                 // token 0 untouched
                if (is_kept && (ids[i] != ids0[i] || w[i] != w0[i])) ++errs;              // kept: id and weight keep
                if (!is_kept) {
                    if (w[i] != 0.0f) ++errs;                                             // dropped: weight exactly 0
                    bool in_kept = false;
                    for (int j = 0; j < n; ++j) if (kept[j] && ids0[j] == ids[i]) in_kept = true;
                    if (!in_kept) ++errs;                                                 // id is one the window named
                }
            }
            if (d1 != ids || ww1 != w) ++errs;                                            // deterministic
            std::printf("  cap %2d      T %d  %s\n", cap, T, errs ? "*** CONTRACT BROKEN ***" : "ok");
            bad += errs ? 1 : 0;
        }
        sycl::free(d_ids, q); sycl::free(d_w, q);
    }
    std::printf(bad ? "ecap_parity: FAIL (%d)\n" : "ecap_parity: PASS\n", bad);
    return bad ? 1 : 0;
}
