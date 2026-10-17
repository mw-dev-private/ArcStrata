// Offline parity test for the SYCL port's CPU prefill share (STRATA_PREFILL_CPU_SHARE): the host-side row arithmetic
// of sycl/src/prefill/prefill.cpp without a GPU.  It replays the ported sequence - candidate pick (fewest tokens
// first, <= MAXT, up to round(share * nstream)), the off/order build with the CPU block last, the slot/src fill, the
// CPU thread's out[] placement and the combine's merge into Dm's tail - and checks that every (token, k) pair reads
// back exactly the row its expert computed: the same weighted sum as a run with every expert on the GPU.
//
//   g++ -std=c++20 -Iinclude -o /tmp/opencode/share_rows_test tests/prefill/cpu_share_rows_test.cpp && run
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "strata/kernels/cpu/expert.hpp"     // MAXT, the candidate gate's constant

using strata::kernels::cpu::MAXT;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

// one synthetic chunk: T tokens, K routed experts each, NE experts total; ids[i*K+k] = expert.
// A "row value" stands for the expert's output row: encode(expert, token) so a wrong row is visible.
static int64_t encode(int e, int t) { return (int64_t) e * 100000 + t; }

static void run_case(int T, int K, int NE, double share, const std::vector<int32_t>& ids,
                     const std::vector<int32_t>& host_res) {
    const int64_t TK = (int64_t) T * K;
    // ---- the port's candidate pick (Edit H) ----
    std::vector<int32_t> cnt((size_t) NE, 0);
    for (int64_t i = 0; i < TK; ++i) ++cnt[(size_t) ids[(size_t) i]];
    std::vector<std::pair<int32_t, int32_t>> cand;   // (tokens, expert)
    int64_t nstream = 0, nres = 0;
    for (int e = 0; e < NE; ++e) {
        const int32_t c = cnt[(size_t) e];
        if (c == 0 || host_res[(size_t) e] >= 0) continue;   // resident: never shared
        ++nstream;
        if (c <= MAXT) cand.emplace_back(c, e);
    }
    for (int e = 0; e < NE; ++e) if (host_res[(size_t) e] >= 0 && cnt[(size_t) e] > 0) ++nres;
    std::sort(cand.begin(), cand.end());
    const size_t take = std::min(cand.size(), (size_t) std::llround(share * (double) nstream));
    std::vector<char> on_cpu((size_t) NE, 0);
    int64_t rows_cpu = 0, n_cpu = 0;
    for (size_t i = 0; i < cand.size() && n_cpu < (int64_t) take; ++i) {
        const int32_t e = cand[i].second;
        on_cpu[(size_t) e] = 1;
        rows_cpu += cand[i].first;
        ++n_cpu;
    }
    // ---- the port's off/order build (Edit I): local first, the CPU block last ----
    std::vector<int32_t> off((size_t) NE + 1, 0);
    {
        int32_t r = 0;
        for (int e = 0; e < NE; ++e)
            if (!on_cpu[(size_t) e]) { off[(size_t) e] = r; r += cnt[(size_t) e]; }
        for (int e = 0; e < NE; ++e)
            if (on_cpu[(size_t) e]) { off[(size_t) e] = r; r += cnt[(size_t) e]; }
        off[(size_t) NE] = r;
    }
    const int64_t rows_local = TK - rows_cpu;
    CHECK(off[(size_t) NE] == TK);   // every routed row got a place
    std::vector<int32_t> fill(off.begin(), off.end() - 1), slot((size_t) TK), src((size_t) TK);
    for (int64_t i = 0; i < TK; ++i) {
        const int32_t e = ids[(size_t) i];
        const int32_t p = fill[(size_t) e]++;
        slot[(size_t) i] = p;
        src[(size_t) p] = (int32_t) (i / K);
    }
    // local experts strictly below the tail, the CPU's experts entirely in it
    for (int e = 0; e < NE; ++e) {
        if (cnt[(size_t) e] == 0) continue;
        if (!on_cpu[(size_t) e]) CHECK(off[(size_t) e] + cnt[(size_t) e] <= rows_local);
        else CHECK(off[(size_t) e] >= rows_local);
    }
    // ---- the experts' rows: GPU path writes Dm[p], the CPU path writes cpu_rows[p - r0c] (Edit J) ----
    const int64_t r0c = TK - rows_cpu;
    std::vector<int64_t> Dm((size_t) TK, -1), cpu_rows((size_t) rows_cpu, -1);
    for (int e = 0; e < NE; ++e) {
        if (cnt[(size_t) e] == 0) continue;
        for (int32_t rr = 0; rr < cnt[(size_t) e]; ++rr) {
            const int64_t p = (int64_t) off[(size_t) e] + rr;
            const int64_t val = encode(e, src[(size_t) p]);
            if (on_cpu[(size_t) e]) cpu_rows[(size_t) (p - r0c)] = val;   // j.out[r] = cpu_rows + (p - r0c) * N
            else Dm[(size_t) p] = val;
        }
    }
    // the CPU thread took exactly the experts marked, none missed
    for (int64_t i = 0; i < rows_cpu; ++i) CHECK(cpu_rows[(size_t) i] != -1);
    // ---- the combine merge (Edit K) and the combine itself: per (token, k) pair read slot's row of Dm ----
    std::copy(cpu_rows.begin(), cpu_rows.end(), Dm.begin() + r0c);   // H2D into Dm's tail
    int64_t checked = 0;
    for (int64_t i = 0; i < TK; ++i) {
        const int64_t p = slot[(size_t) i];
        CHECK(p >= 0 && p < TK);
        CHECK(Dm[(size_t) p] == encode(ids[(size_t) i], (int) (i / K)));   // same expert, same token as before the share
        ++checked;
    }
    // Independent direct-routing weighted reference, without off/slot/group tables.
    for (int t = 0; t < T; ++t) {
        int64_t grouped = 0, direct = 0;
        for (int k = 0; k < K; ++k) {
            const int64_t i = (int64_t) t * K + k;
            const int weight = K - k;
            grouped += weight * Dm[(size_t) slot[(size_t) i]];
            direct += weight * encode(ids[(size_t) i], t);
        }
        CHECK(grouped == direct);
    }
    std::fprintf(stderr, "case T=%d K=%d NE=%d share=%.2f: %lld streamed (%lld resident), %lld experts on the CPU, "
                         "%lld rows in its block - every row maps back\n",
                 T, K, NE, share, (long long) nstream, (long long) nres, (long long) n_cpu, (long long) rows_cpu);
}

int main() {
    // A deterministic model: experts 0..3 heavy (resident); NE-1 heavy streamed (always > MAXT, never shareable);
    // a mid band 4..(NE-9) absorbs the bulk of the routed tokens; the last eight experts (NE-8..NE-2) are each
    // routed by exactly one token - <= MAXT, so they are shareable (real prompts route a long tail of experts
    // that only one or two tokens pick - at 400 tokens the production profile has ~180 touched experts per layer).
    auto make = [](int T, int K, int NE, std::vector<int32_t>* ids, std::vector<int32_t>* res) {
        ids->assign((size_t) T * K, 0);
        res->assign((size_t) NE, -1);
        for (int e = 0; e < 4; ++e) (*res)[(size_t) e] = e;   // resident in the GPU cache: never shared
        const int rare_lo = NE - 8;                            // rare band: one token each, few enough to share
        for (int64_t i = 0; i < T; ++i)
            for (int k = 0; k < K; ++k) {
                const int64_t x = i * K + k;
                if (k == K - 1) { (*ids)[(size_t) x] = (int32_t) (NE - 1); continue; }   // heavy streamed
                if (k < K / 2) { (*ids)[(size_t) x] = (int32_t) (x % 4); continue; }      // heavy resident
                (*ids)[(size_t) x] = (int32_t) (4 + (x % (rare_lo - 4)));                 // mid band
            }
        for (int e = rare_lo; e <= NE - 2; ++e)               // one token each: the share's long tail
            (*ids)[(size_t) ((int64_t) (e % T) * K + 2)] = (int32_t) e;
    };
    std::vector<int32_t> ids, res;

    make(64, 8, 32, &ids, &res);
    run_case(64, 8, 32, 0.0, ids, res);       // share off: bit-identical baseline walk
    run_case(64, 8, 32, 0.5, ids, res);
    run_case(64, 8, 32, 1.0, ids, res);       // everything shareable goes to the CPU

    make(400, 8, 128, &ids, &res);
    run_case(400, 8, 128, 0.4, ids, res);     // the window's ~400-token chunk
    run_case(400, 8, 128, 0.5, ids, res);

    make(2000, 8, 128, &ids, &res);
    run_case(2000, 8, 128, 0.3, ids, res);

    make(400, 10, 128, &ids, &res);
    run_case(400, 10, 128, 0.25, ids, res);
    run_case(400, 10, 128, 0.5, ids, res);
    std::fill(res.begin(), res.end(), 0);
    run_case(400, 10, 128, 1.0, ids, res);  // All resident: no CPU work.
    if (failed) { std::fprintf(stderr, "%d FAILURES\n", failed); return 1; }
    std::printf("cpu_share_rows_test: ok\n");
    return 0;
}
