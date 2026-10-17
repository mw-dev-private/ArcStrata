// src/kernels/qsa_attn_scale_bench.cpp - what limits the prompt path's attention kernel (qsa_decode_attn_batch, the production
// path with int8 KV; 14.10. breakdown: "qsa attn" 2992 ms = 28 % of a 16k prefill)?  Synthetic pools, no model.
// One process times the kernel, called in batches of 32 as prefill.cpp does, over
//   KV format   : int8 (production) and fp16 (no unpacking)
//   selection   : drift (a recent window + older cells that drift ~3 % per query, like a real prompt), same (every query reads the
//                 same first-w cells: all loads hit the caches), rand (w random cells per query: no sharing beyond chance)
//   width       : the full 2,051 cells, then 1,024 / 512 / 128 (the width the kernel reads from steps; fewer cells, same queries)
// and prints ms for the chunk and the same scaled to a 16,383-query layer.  Reading it:
//   time ~ width        -> per-cell cost (compute or memory per cell);  time ~ constant -> fixed cost per query/launch
//   same << rand        -> memory/latency bound (cache hits help);  same ~ rand -> not memory bound
//   fp16 << int8        -> the int8 unpacking costs;  fp16 ~ int8 -> it does not
// Usage: qsa_attn_scale_bench [context=16384] [queries=4096] [reps=5]
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
template <typename T> T* up(const std::vector<T>& h) {
    sycl::queue& q = dpct::get_in_order_queue();
    T* d = (T*) sycl::malloc_device(h.size() * sizeof(T) + 64, q);
    q.memcpy(d, h.data(), h.size() * sizeof(T)).wait();
    return d;
}
uint16_t f2h(float f) {
    sycl::half h = sycl::vec<float, 1>(f).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    return *reinterpret_cast<uint16_t*>(&h);
}

struct Data {
    k::QsaShapes s;
    int64_t ctx, nq, cap;
    k::QsaAttnPools pl;
    float* d_q = nullptr;
    float* d_out = nullptr;
    float* scratch = nullptr;
    int32_t* d_ids = nullptr;
    int32_t* d_steps = nullptr;
    std::vector<int32_t> steps_full;
};

// pools + queries for one KV format (fmt 1 = int8, 0 = fp16)
void make_pools(Data& D, int fmt, int64_t ctx, int64_t nq) {
    D.s = k::qsa_real_shapes();
    D.ctx = ctx; D.nq = nq;
    const int64_t HD = D.s.head_dim, NKV = D.s.n_head_kv, NH = D.s.n_head, PS = D.s.page_size;
    const int64_t pages = (ctx + PS - 1) / PS, rows = pages * NKV * PS;
    std::mt19937 rng(1234 + fmt);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.03f);
    D.pl = k::QsaAttnPools{};
    if (fmt == 1) {
        std::vector<int8_t> kq(rows * HD), vq(rows * HD);
        std::vector<uint16_t> ks(rows * 4), vs(rows * 4);
        for (auto& x : kq) x = (int8_t) code(rng);
        for (auto& x : vq) x = (int8_t) code(rng);
        for (auto& x : ks) x = f2h(sc(rng));
        for (auto& x : vs) x = f2h(sc(rng));
        D.pl.k_q = up(kq); D.pl.v_q = up(vq); D.pl.k_scale = up(ks); D.pl.v_scale = up(vs);
    } else {
        std::vector<uint16_t> kh(rows * HD), vh(rows * HD);
        for (auto& x : kh) x = f2h(nd(rng) * 1.5f);
        for (auto& x : vh) x = f2h(nd(rng));
        D.pl.k_pool = up(kh); D.pl.v_pool = up(vh);
    }
    std::vector<int32_t> table(pages);
    for (int64_t i = 0; i < pages; ++i) table[i] = (int32_t) i;
    std::shuffle(table.begin(), table.end(), rng);   // the page table is a shuffled permutation, as in the parity program
    D.pl.page_table = up(table);
    std::vector<float> q((size_t) (nq * NH * HD));
    for (auto& x : q) x = nd(rng) * 2.0f;
    D.d_q = up(q);
    D.cap = k::qsa_selection_width(ctx, D.s);
    sycl::queue& qu = dpct::get_in_order_queue();
    D.d_out = (float*) sycl::malloc_device((size_t) nq * NH * HD * 4, qu);
    D.scratch = (float*) sycl::malloc_device((size_t) 32 * k::qsa_decode_attn_scratch_floats(D.cap, D.s) * 4, qu);
    D.d_ids = nullptr; D.d_steps = nullptr;
}

// selection pattern: 0 drift, 1 same, 2 rand; returns the full-width ids and steps
void make_selection(Data& D, int pattern, std::vector<int32_t>& ids, std::vector<int32_t>& steps) {
    const int64_t nq = D.nq, ctx = D.ctx, cap = D.cap;
    ids.assign((size_t) (nq * cap), 0); steps.assign((size_t) (nq * k::kStepCount), 0);
    std::mt19937 rng(77 + pattern);
    std::vector<int32_t> old_cells;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t nkv = ctx - nq + i + 1;
        const int64_t w = k::qsa_selection_width(nkv, D.s);
        int32_t* sel = ids.data() + i * cap;
        steps[i * k::kStepCount + k::kStepWidth] = (int32_t) w;
        if (w == nkv) { for (int64_t c = 0; c < w; ++c) sel[c] = (int32_t) c; continue; }
        if (pattern == 1) { for (int64_t c = 0; c < w; ++c) sel[c] = (int32_t) c; continue; }
        if (pattern == 2) {
            std::vector<int32_t> all((size_t) nkv);
            for (int64_t c = 0; c < nkv; ++c) all[c] = (int32_t) c;
            std::shuffle(all.begin(), all.end(), rng);
            std::sort(all.begin(), all.begin() + w);
            std::copy(all.begin(), all.begin() + w, sel);
            continue;
        }
        const int64_t recent = 512, older = w - recent;   // drift: as qsa_prompt_attn_parity.cpp
        if ((int64_t) old_cells.size() != older) {
            std::vector<int32_t> all((size_t) (nkv - recent));
            for (int64_t c = 0; c < nkv - recent; ++c) all[c] = (int32_t) c;
            std::shuffle(all.begin(), all.end(), rng);
            old_cells.assign(all.begin(), all.begin() + older);
        } else {
            std::uniform_int_distribution<int64_t> pick(0, older - 1), any(0, nkv - recent - 1);
            for (int r = 0; r < older / 32; ++r) {
                const int32_t c = (int32_t) any(rng);
                if (std::find(old_cells.begin(), old_cells.end(), c) == old_cells.end()) old_cells[pick(rng)] = c;
            }
        }
        std::vector<int32_t> v(old_cells);
        for (int64_t c = nkv - recent; c < nkv; ++c) v.push_back((int32_t) c);
        std::sort(v.begin(), v.end());
        std::copy(v.begin(), v.end(), sel);
    }
}

double g_chk = 0;
double time_kernel(Data& D, int reps) {
    const int64_t batch = 32, NH = D.s.n_head, HD = D.s.head_dim;
    auto run = [&]() {
        for (int64_t t0 = 0; t0 < D.nq; t0 += batch)
            k::qsa_decode_attn_batch(D.d_q + t0 * NH * HD, D.pl, D.d_ids + t0 * D.cap, D.d_steps + t0 * k::kStepCount, D.cap, D.s,
                                     D.scratch, D.d_out + t0 * NH * HD, std::min(batch, D.nq - t0), nullptr);
    };
    sycl::queue& q = dpct::get_in_order_queue();
    run(); run(); q.wait();   // warm-up (JIT, caches)
    std::vector<double> ms;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        run(); q.wait();
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    // checksum of the output (weighted sum): the variants (STRATA_ATTN_LANECELL / STRATA_ATTN_PRE75) must agree
    std::vector<float> o((size_t) D.nq * NH * HD);
    q.memcpy(o.data(), D.d_out, o.size() * 4).wait();
    double chk = 0;
    for (size_t i = 0; i < o.size(); ++i) chk += (double) o[i] * (double) (1 + (i % 7));
    g_chk = chk;
    return ms[ms.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
    const int64_t ctx = argc > 1 ? std::atoll(argv[1]) : 16384, nq = argc > 2 ? std::atoll(argv[2]) : 4096;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 5;
    sycl::queue& q = dpct::get_in_order_queue();
    std::printf("qsa_attn_scale_bench device=%s ctx=%lld queries=%lld (the last %lld positions) reps=%d; 'layer' = scaled to 16383 queries\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), (long long) ctx, (long long) nq, (long long) nq, reps);
    std::printf("%-5s %-6s %6s %10s %12s %12s %16s\n", "kv", "select", "width", "chunk ms", "layer ms", "us/query", "checksum");
    const char* kvn[] = {"fp16", "int8"};
    const char* pn[] = {"drift", "same", "rand"};
    const int widths[] = {0, 1024, 512, 128};   // 0 = the full width
    for (int fmt : {1, 0}) {
        Data D;
        make_pools(D, fmt, ctx, nq);
        for (int pat : {0, 1, 2}) {
            std::vector<int32_t> ids, steps;
            make_selection(D, pat, ids, steps);
            if (D.d_ids) { sycl::free(D.d_ids, q); sycl::free(D.d_steps, q); }
            D.d_ids = up(ids);
            for (int W : widths) {
                std::vector<int32_t> st = steps;
                int64_t shown = 0;
                for (int64_t i = 0; i < nq; ++i) {
                    int32_t& w = st[i * k::kStepCount + k::kStepWidth];
                    if (W > 0 && w > W) w = W;
                    shown = std::max<int64_t>(shown, w);
                }
                D.d_steps = up(st);
                const double ms = time_kernel(D, reps);
                std::printf("%-5s %-6s %6lld %10.1f %12.1f %12.2f %16.6e\n", kvn[fmt], pn[pat], (long long) shown, ms,
                            ms * 16383.0 / (double) nq, ms * 1000.0 / (double) nq, g_chk);
                std::fflush(stdout);
                sycl::free(D.d_steps, q); D.d_steps = nullptr;
            }
        }
    }
    return 0;
}
