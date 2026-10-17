// bench/gdn_stage_bench.cpp — Zeit je Stufe des Intel-GDN-Prefill-Pfads (Runde 3, Aufgabe 2).
//
// Grund: das Owner-Fenster 14.10. 19:21 mass bei 16k-Prefill 580 ms GDN-Rekurrenz / 36 Schichten
// = 16,1 ms/Schicht, waehrend der Intel-Kern allein (S4-Bench) 5,6 ms braucht — ~10 ms verschwinden
// in der Stage. Die Traffic-Tabelle (STATE.md, Aufgabe 1) sagt fuer den heutigen Stage-Verkehr
// 2,14 GB/Schicht = 3,5 ms (Spitze 608 GB/s) bzw. 4,8 ms (real 450 GB/s). Dieser Bench misst jede
// Stufe einzeln (Median aus 5, wait()-gestoppt, also inkl. Start-Overhead) und schreibt ms + GB/s
// je Zeile — damit entscheidet das Fenster, WELCHE Stufe die Luecke fuellt, nicht der Autor.
//
// Ablauf: synthetische, endliche Eingabedaten (kein Paritaetsanspruch), Scratch wie in der Engine
// (ensure), erst je Stufe 5 Messungen in Pipeline-Reihenfolge, dann Kern allein, dann volles run().
// Der Vendored-Kern wird JIT gebaut — die erste Runde compiliert, der Median aus 5 traegt das.
//
// Bau: CMake-Ziel gdn_stage_bench (nur mit STRATA_INTEL_GDN). Ausfuehren faehrt der Owner:
//   build-sycl/sycl/... wird nicht veraendert; Binary liegt unter build-sycl/bench/.
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "intel_gdn/intel_gdn.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

namespace {
using clk = std::chrono::steady_clock;

struct Step { const char* name; double mb; double ms[5]; };  // mb = Lese+Schreib-Bytes (1e6)

double run5(sycl::queue& q, Step& s, const std::function<void()>& fn) {
  for (int i = 0; i < 5; ++i) {
    const auto a = clk::now();
    fn();
    q.wait();
    const auto b = clk::now();
    s.ms[i] = std::chrono::duration<double, std::milli>(b - a).count();
  }
  std::sort(s.ms, s.ms + 5);
  return s.ms[2];
}

// deterministischer, endlicher Musterwert ~ [-2,2) fuer die Stage-Eingaenge.
float patt(uint64_t i) {
  uint64_t x = i * 0x9E3779B97F4A7C15ull + 0x100000001B3ull;
  x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
  return ((int)(x % 4001) - 2000) / 1000.0f;
}
}  // namespace

int main(int argc, char** argv) {
  const int64_t T = (argc > 1) ? std::atoll(argv[1]) : 16384;
  constexpr int S = 128, HK = 16, HV = 48;
  const int64_t TP = ((T + 63) / 64) * 64;
  constexpr int64_t C = 2 * HK * S + HV * S;         // 10240

  sycl::queue q(dpct::get_in_order_queue().get_device(),
                sycl::property_list{sycl::property::queue::in_order{}});
  const auto dev = q.get_device();
  if (!dev.is_gpu() || dev.get_info<sycl::info::device::vendor_id>() != 0x8086) {
    std::printf("gdn_stage_bench: Intel-GPU noetig (geraet: %s)\n",
                dev.get_info<sycl::info::device::name>().c_str());
    return 2;
  }
  std::printf("gdn_stage_bench device=%s T=%lld TP=%lld S=%d HK=%d HV=%d (synthetische Daten, Median aus 5)\n",
              dev.get_info<sycl::info::device::name>().c_str(), (long long)T, (long long)TP, S, HK, HV);

  strata_intel_gdn::Scratch sc;
  if (!sc.ensure(q, T)) { std::printf("scratch.ensure fehlgeschlagen (T=%lld)\n", (long long)T); return 1; }

  float* hbuf = (float*)sycl::malloc_device((size_t)T * C * 4, q);
  float* ab   = (float*)sycl::malloc_device((size_t)T * 2 * HV * 4, q);
  float* dts  = (float*)sycl::malloc_device(HV * 4, q);
  float* ssm  = (float*)sycl::malloc_device(HV * 4, q);
  float* st   = (float*)sycl::malloc_device((size_t)HV * S * S * 4, q);
  float* y    = (float*)sycl::malloc_device((size_t)T * HV * S * 4, q);
  if (!hbuf || !ab || !dts || !ssm || !st || !y) { std::printf("malloc fehlgeschlagen\n"); return 1; }

  // Fuellung: hbuf/ab Muster; dt positiv, ssm_a negativ (alogp=log(-ssm_a) eindeutig endlich);
  // a in [-2,0] (cumsum fallend, exp endlich), b wie Muster (sigmoid begrenzt); Zustand 0.
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class BGBFillH>(sycl::range<1>((size_t)T * C), [=](sycl::item<1> it) {
      const uint64_t i = it.get_id(0); hbuf[i] = patt(i * 2654435761ull + 17);
    });
  });
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class BGBFillAB>(sycl::range<1>((size_t)T * 2 * HV), [=](sycl::item<1> it) {
      const uint64_t i = it.get_id(0);
      ab[i] = (i % 2 == 0) ? -std::fabs(patt(i)) * 0.03f : patt(i * 7 + 3);
    });
  });
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class BGBFillHead>(sycl::range<1>(HV), [=](sycl::item<1> it) {
      const int h = (int)it.get_id(0);
      dts[h] = 0.05f + 0.01f * h;      // dt positiv klein
      ssm[h] = -(0.5f + 0.05f * h);    // ssm_a = -exp(A_log) < 0
    });
  });
  q.submit([&](sycl::handler& cgh) { cgh.parallel_for<class BGBFillSt>(sycl::range<1>((size_t)HV * S * S),
    [=](sycl::item<1> it) { st[it.get_id(0)] = 0.0f; }); });
  q.wait();

  // Byte-Budgets pro Stufe (Lesen+Schreiben), wie STATE.md Aufgabe 1.
  const double B_QKV = ((double)T * C * 4 + (double)TP * (2.0 * HK * S + HV * S) * 2) / 1e6;
  const double B_GATE = ((double)T * 2 * HV * 4 + (double)2 * HV * TP * 4 + HV * (4 + 4 + 2)) / 1e6;
  const double B_ST = (2.0 * HV * S * S * 4) / 1e6;
  const double B_MEM = ((double)HV * (T + 63) * (64 + 128 + 128) * 2) / 1e6;
  // Kern orientierungsweise: q/k/v-lesen (k zweimal: compute_A+WU), v einmal, q einmal,
  // A schreiben+inverse+2x lesen, w/u schreiben je + lesen je, istate r+w, iout schreiben, a r+w, b 2x lesen.
  const double B_KERN = ((double)TP * HK * S * 2 * 2 + (double)TP * HV * S * 2      // q,k(2x),v
      + 4.0 * HV * TP * 64 * 2 + 2.0 * 2 * HV * TP * 128 * 2                       // A 4x, w+u je r+w
      + 2.0 * HV * S * S * 4 + (double)TP * HV * S * 2                             // istate r+w, iout w
      + (double)HV * TP * 4 * 6 + (double)HV * TP * 4 * 3) / 1e6;                  // a ~6x, b ~3x
  const double B_OUT = ((double)T * HV * S * 2 + (double)T * HV * S * 4) / 1e6;

  Step steps[] = {
    {"stage_qkv", B_QKV, {}}, {"stage_gates", B_GATE, {}}, {"state_in", B_ST, {}},
    {"memset_ws", B_MEM, {}}, {"kernel_only", B_KERN, {}}, {"out_write", B_OUT, {}},
    {"state_out", B_ST, {}},
  };

  // Mesihenfolge = Pipelinereihefolge; jede Stufe schreibt ihre Ziel-Puffer, spatere Stufen sehen
  // gueltige Daten. kernel_only veraendert istate (Delta-Rekursion) — numerisch endlich, Timing unbeuert.
  double sum = 0.0;
  auto step = [&](int i, const std::function<void()>& fn) {
    const double m = run5(q, steps[i], fn);
    sum += m;
    std::printf("  %-11s %9.1f MB  %8.3f ms  %7.1f GB/s   [%s]\n", steps[i].name, steps[i].mb, m,
                steps[i].mb / m,
                i == 4 ? "Bytes orientiert, nicht exakt" :
                (i == 3 ? "nur mit STRATA_GDN_WS_MEMSET=1 in run() — Audit: entbehrlich" : "Bytes exakt"));
  };
  step(0, [&] { strata_intel_gdn::stage_qkv(q, hbuf, sc.qpad, sc.kpad, sc.vpad, T, TP, S, HK, HV); });
  step(1, [&] { strata_intel_gdn::stage_gates(q, ab, sc.ia, sc.ib, dts, ssm, sc.dt16, sc.alogp,
                                              sc.qsl, sc.ci, sc.hinit, T, TP, HK, HV); });
  step(2, [&] { strata_intel_gdn::state_in(q, st, sc.istate, S, HK, HV); });
  step(3, [&] { strata_intel_gdn::memset_ws(q, sc, T); });
  step(4, [&] { strata_intel_gdn::kernel_only(q, sc, T, S, HK, HV); });
  step(5, [&] { strata_intel_gdn::out_write(q, sc.iout, y, T, S, HK, HV); });
  step(6, [&] { strata_intel_gdn::state_out(q, sc.istate, st, S, HK, HV); });
  std::printf("  %-11s %9.1f MB  %8.3f ms  (Summe der Stufen ueinander)\n", "SUMME", 0.0, sum);

  // Vergleich: volles run() (Engine-Pfad, memset gem. STRATA_GDN_WS_MEMSET).
  Step full = {"run_full", 0.0, {}};
  const double fm = run5(q, full, [&] {
    strata_intel_gdn::run(q, sc, hbuf, ab, dts, ssm, st, y, T, S, HK, HV);
  });
  std::printf("  %-11s %9.1f MB  %8.3f ms  (run() komplett, inkl. Start-Overhead der Kette)\n", "run()", 0.0, fm);
  std::printf("  Referenz: Intel-Kern allein S4-Bench (T=16383, inkl. torch::zeros-Protokoll): ~5,6 ms\n");

  // Controlline unters Fenster (Nachlese 14.10.): die absolute Endlichkeitspruefung trennt nicht
  // zwischen "synthetische Gates detonieren" und "Kern liesst Ungeschriebenes" — sie riss im Fenster
  // komplett (790528 = alle geprueften Werte), obwohl die Paritaet grin war. Beweis des memset-Audits
  // ist stattdessen der Vergleich zweierLaufe mit UNTERSCHIEDLicher Startfuellung von A/w/u:
  // Lauf 1 startet vergiftet (fp16-NaN), Lauf 2 mit den Altresten aus Lauf 1 — bitidentische y bei
  // identischem Anfangszustand == der Inhalt von A/w/u vor dem Kernstart ist gleichgueltig.
  {
    float* st0 = static_cast<float*>(sycl::malloc_device(sizeof(float) * HV * S * S, q));
    q.memcpy(st0, st, sizeof(float) * HV * S * S).wait();
    const size_t ny = (size_t)T * HV * S;
    std::vector<float> y1(ny), y2(ny);
    strata_intel_gdn::poison_ws(q, sc, T);
    strata_intel_gdn::run(q, sc, hbuf, ab, dts, ssm, st, y, T, S, HK, HV);
    q.wait();
    q.memcpy(y1.data(), y, sizeof(float) * ny).wait();
    q.memcpy(st, st0, sizeof(float) * HV * S * S).wait();   // selber Anfangszustand wie Lauf 1
    strata_intel_gdn::run(q, sc, hbuf, ab, dts, ssm, st, y, T, S, HK, HV);  // Startfuellung: Altreste
    q.wait();
    q.memcpy(y2.data(), y, sizeof(float) * ny).wait();
    size_t diff = 0, bad = 0;
    for (size_t i = 0; i < ny; ++i) {
      if (std::memcmp(&y1[i], &y2[i], 4) != 0) ++diff;      // bitvergleichend: identische NaN-Muster zahlen nicht als Diff
      if (!(y2[i] == y2[i]) || y2[i] > 1e30f || y2[i] < -1e30f) ++bad;
    }
    std::printf("  Poison-A/B (y1[NaN-Start] vs y2[Altreste]): %s (%zu/%zu bitverschieden)\n",
                diff ? "DIFFERENZ — Audit false, WS_MEMSET noetig!" : "IDENTISCH — Nullfuellung nachweislich entbehrlich", diff, ny);
    std::printf("  Endlichkeit y (nur informativ — synthetische Gates ohne reale Zerfallsstruktur koennen detonieren): %zu nicht-endliche\n", bad);
    sycl::free(st0, q);
  }

  sycl::free(hbuf, q); sycl::free(ab, q); sycl::free(dts, q); sycl::free(ssm, q);
  sycl::free(st, q); sycl::free(y, q);
  sc.free_all();
  return 0;
}
