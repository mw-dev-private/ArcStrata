// sycl/src/kernels/mmvq_q6k_fast_parity.cpp - Q6_K decode mmvq: the FAST unpack (STRATA_MMVQ_Q6K_FAST) against the wide kernel.
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 mmvq_q6k_fast_parity [--bench]
// Parity gate: every output float must be bit-identical (0 mismatches). --bench also prints us per call, off / on.
// Refuses to run without ONEAPI_DEVICE_SELECTOR: level_zero:0 is the production card.
#include "strata/kernels/native_mmvq.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
int main(int argc, char** argv) {
    if (std::getenv("ONEAPI_DEVICE_SELECTOR") == nullptr) { std::fprintf(stderr, "set ONEAPI_DEVICE_SELECTOR (never the production card)\n"); return 2; }
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    sycl::queue* s = &dpct::get_in_order_queue();
    std::printf("device: %s\n", s->get_device().get_info<sycl::info::device::name>().c_str());
    const int shapes[][2] = {{2560, 10240}, {2560, 12288}, {6144, 2560}, {2560, 640}, {2560, 2560}, {256, 37}, {512, 300}, {768, 1}};
    std::mt19937 rng(7);
    long total_mismatch = 0;
    for (auto& sh : shapes) {
        const int n_in = sh[0], n_out = sh[1];
        const size_t wbytes = (size_t) n_out * (n_in / 256) * 210;
        std::vector<uint8_t> hw(wbytes);
        for (size_t i = 0; i < wbytes; ++i) hw[i] = (uint8_t) rng();
        for (size_t b = 0; b < wbytes / 210; ++b) {   // d = a finite fp16 (exponent 0x2800..0x2fff) at offset 208
            const uint16_t d = 0x2800 + (rng() & 0x07ff);
            std::memcpy(&hw[b * 210 + 208], &d, 2);
        }
        void* w = sycl::malloc_device(wbytes, *s);
        s->memcpy(w, hw.data(), wbytes).wait();
        for (int nc = 1; nc <= 8; ++nc) {
            std::vector<float> hx((size_t) nc * n_in);
            for (auto& v : hx) v = (float) (rng() % 2001) / 1000.f - 1.f;
            float* x = sycl::malloc_device<float>(hx.size(), *s);
            void* xq = sycl::malloc_device((size_t) nc * (n_in / 32) * 36, *s);
            float* y0 = sycl::malloc_device<float>((size_t) nc * n_out, *s);
            float* y1 = sycl::malloc_device<float>((size_t) nc * n_out, *s);
            s->memcpy(x, hx.data(), hx.size() * 4).wait();
            strata::kernels::native_quantize_q8_1(x, xq, n_in, nc, s);
            strata::kernels::native_mmvq_set_q6k_fast(false);
            strata::kernels::native_q6_k_mmvq(w, xq, y0, n_in, n_out, nc, s);
            strata::kernels::native_mmvq_set_q6k_fast(true);
            strata::kernels::native_q6_k_mmvq(w, xq, y1, n_in, n_out, nc, s);
            s->wait();
            std::vector<float> a((size_t) nc * n_out), b(a.size());
            s->memcpy(a.data(), y0, a.size() * 4).wait(); s->memcpy(b.data(), y1, b.size() * 4).wait();
            long mm = 0; for (size_t i = 0; i < a.size(); ++i) if (std::memcmp(&a[i], &b[i], 4) != 0) ++mm;
            total_mismatch += mm;
            double us[2] = {0, 0};
            if (bench && n_out >= 600) for (int on = 0; on < 2; ++on) {
                strata::kernels::native_mmvq_set_q6k_fast(on != 0);
                auto run = [&] { strata::kernels::native_q6_k_mmvq(w, xq, on ? y1 : y0, n_in, n_out, nc, s); };
                for (int i = 0; i < 50; ++i) run();
                s->wait();
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 300; ++i) run();
                s->wait();
                us[on] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 300;
            }
            if (mm || (bench && n_out >= 600 && (nc == 1 || nc == 2 || nc == 6)))
                std::printf("q6_k %5d x %5d cols %d  mismatches %ld/%zu  base %.1f us  fast %.1f us\n", n_in, n_out, nc, mm, a.size(), us[0], us[1]);
            sycl::free(x, *s); sycl::free(xq, *s); sycl::free(y0, *s); sycl::free(y1, *s);
        }
        sycl::free(w, *s);
    }
    std::printf("TOTAL mismatches: %ld\n", total_mismatch);
    return total_mismatch ? 1 : 0;
}
