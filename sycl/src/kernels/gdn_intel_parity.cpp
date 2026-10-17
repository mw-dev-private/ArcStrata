// src/kernels/gdn_intel_parity.cpp - Paritaet des opt-in Intel-Delta-Arms (STRATA_GDN_INTEL, README_gdn.md).
//
// Zwei Ebenen, weil der Arm drei Dinge gleichzeitig richtig machen muss:
//   (a) die Kopfpaarung Strata-Slot <-> Intel-Kopf (Modulo vs Block, S4: beides liefert wohlgeformte
//       Ausgaben, nur eine ist die richtige),
//   (b) das Zustandslayout [k][h][v] <-> [hv][v][k] (ein Vertipper: falsche Zahlen, keine Warnung),
//   (c) die numerische Aequivalenz der chunked Delta-Rekursion gegen die etablierte Xe-Variante.
//
// --selftest prueft (a) und (b) als Wirtsrechnung: Bijektivitaet der Permutation, Konsistenz der
// q-Kopf-Zuordnung beider Konventionen, und der Zustands-Umbau als Hin-und-Zurueck-Rundlauf mit
// kodierten Koordinaten. Das ist reine Indexrechnung, laeuft ohne GPU. Die in den Stage-Durchzuegen
// verwendeten Helfer perm_h2i/perm_i2h liegen dafuer in intel_gdn.hpp (dieselbe Funktion, kein Abklatsch).
//
// GPU-Modus (fahert im Fenster): liest die fp64-Referenz aus ~/strata/istrata-s4/ref/gdn_ref_T256.bin
// (Format GDNREFv1: magic, {T,hk,hv,S}, FNV-crc, dann yb, ym, q, k, v, ab, dt, alog als fp32; ym traegt
// die modulo-Zuordnung in Strata-Slot-Nummerierung und die Modellskalierung q*rsqrt(S), s4 README).
// Aus denselben Rohwerten: hbuf = eps-l2norm(q)|eps-l2norm(k)|v (dicht, Strata-Layout) und die Rohgate
// ab/dt/ssm_a=-exp(alog). Arm S faehrt die etablierte Xe-Kette (gdn_gates + gdn_xe_launch), Arm I die
// Intel-Kette (strata_intel_gdn::run); beides Anfangszustand 0 wie in der Referenz. Verglichen wird
// y gegen ym und der Endzustand Arm S gegen Arm I (eine Referenz fuer den Zustand existiert nicht;
// derselbe Fehler in (b) wuerde beide Arme nicht gleichmachen, sondern weit auseinandertreiben).
//
// --expect <datei> (wirtseitig, ohne GPU) schreibt den erwarteten Inhalt aller Stage-Puffer
// (qpad,kpad,vpad,ia,ib,dt16,alogp,istate) als GSTAGEv1-Datei; im GPU-Lauf vergleicht --stagecheck
// (Standard an) dieselbe Wirt-Erwartung Bit fuer Bit gegen die Rueckkopie der echten Stage-Puffer
// (run(stage_only=true)). Das trennt Stage-Fehler von Kernfehler — Fenster 14.10.: Intel-Arm
// max_abs 0,03937 mit nicht-finite Werten, waehrend der S4-Bench mit Host-prep PASS zeigte.
//
// Die Schwellwerte unten sind Diagnoese, kein Beweis: S4-Messbasis (T=16383, v-permutiert) zeigte
// Arm-vs-Referenz max_abs ~0,07-0,08 bei |ref|_mittel ~1,0. Endgueltige Annahme trifft das Fenster.
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "strata/prefill/kernels.hpp"
#include "strata/prefill/gdn_xe.hpp"
#include "intel_gdn/intel_gdn.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int S = 128, HK = 16, HV = 48, C = 10240;
constexpr float NORM_EPS = 1e-6f;   // eps-Floor auf der Quadratsumme (beider Seiten, s4-Bench L45)

// ---------------- selftest: Wirtsseite, keine GPU ----------------

int selftest() {
    const int rep = HV / HK;
    int fails = 0;
    auto fail = [&](const char* what) { std::printf("SELFTEST FAIL: %s\n", what); ++fails; };

    // 1. Permutation bijektiv, Involution, und die q-Kopf-Zuordnung stimmt in beiden Konventionen ueberein:
    //    Strata slot h liest q-Kopf h % HK, Intel-Kopf hv liest hv / rep - verknuepft perm genau diese.
    std::vector<int> seen(HV, 0);
    for (int h = 0; h < HV; ++h) {
        const int hv = strata_intel_gdn::perm_h2i(h, HK, rep);
        if (hv < 0 || hv >= HV) fail("perm_h2i ausserhalb");
        else if (strata_intel_gdn::perm_i2h(hv, HK, rep) != h) fail("perm ist keine Involution");
        else if (++seen[hv] != 1) fail("perm_h2i nicht bijektiv");
        if (strata_intel_gdn::perm_i2h(hv, HK, rep) % HK != hv / rep) fail("q-Kopf der Paarung inkonsistent");
    }
    for (int hv = 0; hv < HV; ++hv)
        if (seen[hv] != 1) fail("perm_i2h trifft nicht jeden Intel-Kopf");

    // 2. Zustands-Umbau [k][h][v] -> [hv][v][k] (state_in/state_out) als Rundlauf mit kodierten Werten.
    std::vector<float> st((size_t)S * HV * S), ist((size_t)HV * S * S), st2(st.size());
    for (int k = 0; k < S; ++k)
        for (int h = 0; h < HV; ++h)
            for (int v = 0; v < S; ++v)
                st[(size_t)k * HV * S + (size_t)h * S + v] = (float)(k * 100000 + h * 1000 + v);
    for (int hv = 0; hv < HV; ++hv) {                 // state_in nachgebaut (intel_gdn_stage.dp.cpp)
        const int h = strata_intel_gdn::perm_i2h(hv, HK, rep);
        for (int v = 0; v < S; ++v)
            for (int k = 0; k < S; ++k)
                ist[(size_t)hv * S * S + (size_t)v * S + k] = st[(size_t)k * HV * S + (size_t)h * S + v];
    }
    for (int hv = 0; hv < HV; ++hv) {                 // state_out nachgebaut
        const int h = strata_intel_gdn::perm_i2h(hv, HK, rep);
        for (int v = 0; v < S; ++v)
            for (int k = 0; k < S; ++k)
                st2[(size_t)k * HV * S + (size_t)h * S + v] = ist[(size_t)hv * S * S + (size_t)v * S + k];
    }
    if (st2 != st) fail("Zustands-Umbau Hin-und-Zurueck nicht identisch");

    // 3. Semantik-Stichprobe: ein koordinatenkodierter Wert muss im Intel-Layout an der
    //   erwarteten Zelle liegen (finge eine vertauschte v/k-Achse, die im Rundlauf unsichtbar bleibt,
    //    wenn in und out denselben Vertipper tragen).
    //    Richtung ist am vendored Kern festgemacht: gdn_kernels_xe2.hpp L972-975 zeigt den Kopf
    //    (v_head_id * head_v_dim * head_k_dim), L1029-1031 baut die S-Matrix mit make_shape(head_v_dim,
    //    head_k_dim) und make_stride(head_k_dim, _1) — Zeile = v, Spalte = k, k-kontinuierlich.
    //    Also Zelle (hv, v, k) = hv*S*S + v*S + k, genau das baut state_in.
    const int k0 = 7, h0 = 20, v0 = 101;
    const int hv0 = strata_intel_gdn::perm_h2i(h0, HK, rep);
    if (ist[(size_t)hv0 * S * S + (size_t)v0 * S + k0] != st[(size_t)k0 * HV * S + (size_t)h0 * S + v0])
        fail("Zustandszelle liegt nicht im erwarteten Layout");

    // 4. Ausgabe-Rueckschreibeschleife out_write (iout [T,HV,S] Intel-Nummerierung -> y Strata-Slots):
    //    dieselbe Permutation wie die Eingangsseite, also muss y[slot h] genau iout[perm_h2i(h)] treffen -
    //    probiert an allen Koepfen mit kodierten Werten (T=1 reicht fuer die Indexrechnung).
    std::vector<float> iout((size_t)HV * S), y((size_t)HV * S);
    for (int hv = 0; hv < HV; ++hv)
        for (int d = 0; d < S; ++d) iout[(size_t)hv * S + d] = (float)(hv * 1000 + d);
    for (int h = 0; h < HV; ++h) {
        const int hv = strata_intel_gdn::perm_h2i(h, HK, rep);
        for (int d = 0; d < S; ++d) y[(size_t)h * S + d] = iout[(size_t)hv * S + d];
    }
    for (int h = 0; h < HV; ++h)
        for (int d = 0; d < S; ++d) {
            const int hv = strata_intel_gdn::perm_h2i(h, HK, rep);
            if (y[(size_t)h * S + d] != (float)(hv * 1000 + d)) { fail("out_write traf falschen Intel-Kopf"); h = HV; break; }
        }

    std::printf("gdn_intel_parity selftest %s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}

// ---------------- GPU-Modus: beide Arme gegen die fp64-Referenz ----------------

uint64_t fnv_bytes(const void* p, size_t n) {   // crc des s4-Bench nachgebaut (FNV-1a)
    uint64_t crc = 1469598103934665603ull;
    const unsigned char* b = (const unsigned char*)p;
    for (size_t i = 0; i < n; ++i) { crc ^= b[i]; crc *= 1099511628211ull; }
    return crc;
}

struct Ref {
    int64_t T; int hk, hv;
    std::vector<float> yb, ym, q, k, v, ab, dt, alog;
    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "ref: %s nicht lesbar\n", path.c_str()); return false; }
        char magic[8]; int32_t hdr[4]; uint64_t crc;
        bool ok = std::fread(magic, 1, 8, f) == 8 && std::fread(hdr, 4, 4, f) == 4 && std::fread(&crc, 8, 1, f) == 1;
        if (!ok || std::memcmp(magic, "GDNREFv1", 8) != 0) { std::fclose(f); std::fprintf(stderr, "ref: magic fehlt\n"); return false; }
        T = hdr[0]; hk = hdr[1]; hv = hdr[2];
        if (hdr[3] != S || hk != HK || hv != HV) {
            std::fprintf(stderr, "ref: Form T=%d hk=%d hv=%d S=%d passt nicht zur Engine (16/48/128)\n", hdr[0], hk, hv, hdr[3]);
            std::fclose(f); return false;
        }
        for (auto* pv : {&yb, &ym}) pv->resize((size_t)T * HV * S);
        for (auto* pv : {&q, &k}) pv->resize((size_t)T * HK * S);
        for (auto* pv : {&v}) pv->resize((size_t)T * HV * S);
        ab.resize((size_t)T * 2 * HV); dt.resize(HV); alog.resize(HV);
        for (auto* pv : {&yb, &ym, &q, &k, &v, &ab, &dt, &alog})
            if (std::fread(pv->data(), 4, pv->size(), f) != pv->size()) { std::fclose(f); std::fprintf(stderr, "ref: abgeschnitten\n"); return false; }
        std::fclose(f);
        uint64_t mine = 1469598103934665603ull;   // FNV-Startwert wie im s4-Bench (war 0: crc mismatch bei jedem Lauf, 14.10.)
        for (auto* pv : {&q, &k, &v, &ab, &dt, &alog}) { mine ^= fnv_bytes(pv->data(), pv->size() * 4); mine *= 1099511628211ull; }
        if (mine != crc) { std::fprintf(stderr, "ref: crc mismatch (Datei %016llx vs %016llx)\n",
                                        (unsigned long long)crc, (unsigned long long)mine); return false; }
        return true;
    }
};

struct Stats { double max_abs = 0, sum_abs = 0, scale = 0; bool nan = false; };
Stats compare(const std::vector<float>& a, const std::vector<float>& b) {
    Stats s;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i])) s.nan = true;
        const double d = std::fabs((double)a[i] - (double)b[i]);
        s.max_abs = std::max(s.max_abs, d); s.sum_abs += d; s.scale += std::fabs((double)b[i]);
    }
    return s;
}

// ---------------- Stage-Erwartung: Wirt-Emulation der Stage-Kerne ----------------
// Spiegelung der Stage-Durchzuege (intel_gdn_stage.dp.cpp) auf der CPU, Rezept identisch zum
// s4-Bench (L279-305: v/a/b/dt/A_log per perm_i2h nach Intel-Nummerierung, a roh, b sigmoid,
// Polsterzeilen 0). Der GPU-Lauf mit stage_only=true kopiert die echten Stage-Puffer zurueck,
// verglichen wird Bit fuer Bit (fp16) bzw. bis 1 ulp (fp32: libdevice gegen libm).
// --expect <datei> schreibt diese Erwartung als GSTAGEv1-Datei (rein wirtsseitig, ohne GPU).

struct StageExp {
    int64_t TP = 0;
    std::vector<uint16_t> qpad, kpad, vpad, dt16;   // rohe fp16-Bitmuster
    std::vector<float> ia, ib, alogp, istate;       // fp32
    std::vector<int32_t> qsl{0, 0}; int32_t ci = 0; uint8_t hinit = 1;
};

uint16_t half_bits(float x) { sycl::half h(x); uint16_t u; std::memcpy(&u, &h, 2); return u; }

StageExp stage_expect(const std::vector<float>& hbuf, const std::vector<float>& ab,
                      const std::vector<float>& dt, const std::vector<float>& ssm_a,
                      const std::vector<float>& state, int64_t T) {
    StageExp e;
    const int64_t TP = e.TP = ((T + 63) / 64) * 64;
    const int64_t HKS = HK * S, HVS = HV * S, C = 2 * HKS + HVS;
    const int rep = HV / HK;
    const float qsc = 1.0f / std::sqrt((float)S);
    e.qpad.assign((size_t)TP * HKS, 0); e.kpad.assign((size_t)TP * HKS, 0);
    e.vpad.assign((size_t)TP * HVS, 0);
    for (int64_t t = 0; t < TP; ++t)
        for (int64_t c = 0; c < C; ++c) {
            const float x = (t < T) ? hbuf[(size_t)t * C + c] : 0.0f;
            if (c < HKS) e.qpad[(size_t)t * HKS + c] = half_bits(x * qsc);
            else if (c < 2 * HKS) e.kpad[(size_t)t * HKS + (c - HKS)] = half_bits(x);
            else {
                const int slot = (int)((c - 2 * HKS) / S), d = (int)((c - 2 * HKS) % S);
                e.vpad[(size_t)t * HVS + (size_t)strata_intel_gdn::perm_h2i(slot, HK, rep) * S + d] = half_bits(x);
            }
        }
    e.ia.assign((size_t)HV * TP, 0.f); e.ib.assign((size_t)HV * TP, 0.f);
    for (int64_t t = 0; t < TP; ++t)
        for (int hv = 0; hv < HV; ++hv) {
            const int h = strata_intel_gdn::perm_i2h(hv, HK, rep);
            if (t < T) {
                e.ia[(size_t)hv * TP + t] = ab[(size_t)t * 2 * HV + h];
                e.ib[(size_t)hv * TP + t] = 1.0f / (1.0f + std::exp(-ab[(size_t)t * 2 * HV + HV + h]));
            }
        }
    e.dt16.assign(HV, 0); e.alogp.assign(HV, 0.f);
    for (int hv = 0; hv < HV; ++hv) {
        const int h = strata_intel_gdn::perm_i2h(hv, HK, rep);
        e.dt16[hv] = half_bits(dt[h]);
        e.alogp[hv] = (ssm_a[h] < 0.0f) ? std::log(-ssm_a[h]) : -1.0e30f;
    }
    e.istate.assign((size_t)HV * S * S, 0.f);
    for (int hv = 0; hv < HV; ++hv) {
        const int h = strata_intel_gdn::perm_i2h(hv, HK, rep);
        for (int v = 0; v < S; ++v)
            for (int k = 0; k < S; ++k)
                e.istate[(size_t)hv * S * S + (size_t)v * S + k] = state[(size_t)k * HV * S + (size_t)h * S + v];
    }
    e.qsl = {0, (int32_t)T}; e.ci = 0; e.hinit = 1;
    return e;
}

bool write_expect(const std::string& path, const StageExp& e, int64_t T) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "expect: %s nicht schreibbar\n", path.c_str()); return false; }
    const char magic[9] = "GSTAGEv1"; int32_t hdr[4] = {(int32_t)T, HK, HV, S};
    std::fwrite(magic, 1, 8, f); std::fwrite(hdr, 4, 4, f);
    std::fwrite(e.qpad.data(), 2, e.qpad.size(), f); std::fwrite(e.kpad.data(), 2, e.kpad.size(), f);
    std::fwrite(e.vpad.data(), 2, e.vpad.size(), f);
    std::fwrite(e.ia.data(), 4, e.ia.size(), f); std::fwrite(e.ib.data(), 4, e.ib.size(), f);
    std::fwrite(e.dt16.data(), 2, e.dt16.size(), f); std::fwrite(e.alogp.data(), 4, e.alogp.size(), f);
    std::fwrite(e.istate.data(), 4, e.istate.size(), f);
    std::fclose(f);
    return true;
}

// Vergleich Rueckkopie gegen Erwartung. fp16: Bitvergleich; fp32: exakt oder <=1ulp (libdevice-
// Divergenz erlaubt, groesseres meldet). Erstfuendige Abweichungen werden mit Koordinaten gedruckt.
struct BufDiff { const char* name; uint64_t diff = 0, near = 0; };

void cmp_bits(BufDiff& d, const std::vector<uint16_t>& got, const std::vector<uint16_t>& exp,
              int64_t cols, int shown, int64_t TP) {
    for (size_t i = 0; i < exp.size(); ++i)
        if (got[i] != exp[i]) {
            // 1 fp16-Einheit gleichen Vorzeichens (Rundung Wirt/Karte, meist Subnormale ~5e-5) ist kein Fehler (Fenster 14.10. 19:21)
            const int ga = got[i] & 0x7fff, ea = exp[i] & 0x7fff;
            if ((got[i] & 0x8000) == (exp[i] & 0x8000) && std::abs(ga - ea) <= 1) { ++d.near; continue; }
            ++d.diff;
            if (shown < 5) {
                auto hf = [](uint16_t u) { sycl::half h; std::memcpy(&h, &u, 2); return (float)h; };
                std::printf("    %s[%llu] (t=%lld,c=%lld): GPU=%.6g Erw=%.6g\n", d.name, (unsigned long long)i,
                            (long long)(i / (uint64_t)cols), (long long)(i % (uint64_t)cols),
                            hf(got[i]), hf(exp[i]));
                ++shown;
            }
            (void)TP;
        }
}

void cmp_f32(BufDiff& d, const std::vector<float>& got, const std::vector<float>& exp,
             int64_t cols, int shown) {
    for (size_t i = 0; i < exp.size(); ++i) {
        if (got[i] == exp[i]) continue;
        const double rel = std::fabs((double)got[i] - (double)exp[i]) / std::max(1e-30, std::fabs((double)exp[i]));
        if (rel <= 1e-6) { ++d.near; continue; }
        ++d.diff;
        if (shown < 5) {
            std::printf("    %s[%llu] (hv=%lld,rest=%lld): GPU=%.9g Erw=%.9g\n", d.name, (unsigned long long)i,
                        (long long)(i / (uint64_t)cols), (long long)(i % (uint64_t)cols), got[i], exp[i]);
            ++shown;
        }
    }
}

// Rueckkopie der Stage-Puffer nach run(stage_only=true) und Vergleich gegen die Erwartung.
// false = harter Unterschied (Paritaet kann nicht grin).
bool stagecheck(sycl::queue& q, strata_intel_gdn::Scratch& sc, const StageExp& e, int64_t T) {
    const int64_t TP = e.TP, HKS = HK * S, HVS = HV * S;
    std::vector<uint16_t> qpad((size_t)TP * HKS), kpad((size_t)TP * HKS), vpad((size_t)TP * HVS), dt16(HV);
    std::vector<float> ia((size_t)HV * TP), ib((size_t)HV * TP), alogp(HV), istate((size_t)HV * S * S);
    std::vector<int32_t> qsl(2); int32_t ci = -1; uint8_t hinit = 0;
    q.memcpy(qpad.data(), sc.qpad, qpad.size() * 2).wait_and_throw();
    q.memcpy(kpad.data(), sc.kpad, kpad.size() * 2).wait_and_throw();
    q.memcpy(vpad.data(), sc.vpad, vpad.size() * 2).wait_and_throw();
    q.memcpy(ia.data(), sc.ia, ia.size() * 4).wait_and_throw();
    q.memcpy(ib.data(), sc.ib, ib.size() * 4).wait_and_throw();
    q.memcpy(dt16.data(), sc.dt16, dt16.size() * 2).wait_and_throw();
    q.memcpy(alogp.data(), sc.alogp, alogp.size() * 4).wait_and_throw();
    q.memcpy(istate.data(), sc.istate, istate.size() * 4).wait_and_throw();
    q.memcpy(qsl.data(), sc.qsl, 8).wait_and_throw();
    q.memcpy(&ci, sc.ci, 4).wait_and_throw();
    q.memcpy(&hinit, sc.hinit, 1).wait_and_throw();

    BufDiff dq{"qpad"}, dk{"kpad"}, dv{"vpad"}, dia{"ia"}, dib{"ib"}, ddt{"dt16"},
          dal{"alogp"}, dis{"istate"};
    cmp_bits(dq, qpad, e.qpad, HKS, 0, TP);
    cmp_bits(dk, kpad, e.kpad, HKS, 0, TP);
    cmp_bits(dv, vpad, e.vpad, HVS, 0, TP);
    cmp_f32(dia, ia, e.ia, TP, 0);
    cmp_f32(dib, ib, e.ib, TP, 0);
    cmp_bits(ddt, dt16, e.dt16, 1, 0, TP);
    cmp_f32(dal, alogp, e.alogp, 1, 0);
    cmp_f32(dis, istate, e.istate, S * S, 0);
    bool hard = false;
    for (auto& d : {dq, dk, dv, dia, dib, ddt, dal, dis})
        if (d.diff) { std::printf("  stage %-7s: %llu ABWEICHUNGEN", d.name, (unsigned long long)d.diff);
                      if (d.near) std::printf(" (+%llu <=1ulp)", (unsigned long long)d.near);
                      std::printf("\n"); hard = true; }
        else if (d.near) std::printf("  stage %-7s: bitgleich, %llu <=1ulp (libdevice/Rundung)\n", d.name, (unsigned long long)d.near);
        else std::printf("  stage %-7s: bitgleich\n", d.name);
    if (qsl[0] != e.qsl[0] || qsl[1] != e.qsl[1] || ci != e.ci || hinit != e.hinit) {
        std::printf("  ctrl qsl={%d,%d} ci=%d hinit=%d -- erwartet {0,%lld} 0 1\n",
                    qsl[0], qsl[1], ci, hinit, (long long)T);
        hard = true;
    } else std::printf("  ctrl qsl/ci/hinit: korrekt\n");
    return !hard;
}

// ---------------- GPU-Modus: beide Arme gegen die fp64-Referenz ----------------

int gpu_case(const std::string& ref_path, bool do_stagecheck) {
    Ref ref;
    if (!ref.load(ref_path)) return 2;
    const int64_t T = ref.T;
    sycl::queue q(dpct::get_in_order_queue().get_device(), sycl::property_list{sycl::property::queue::in_order{}});
    const auto dev = q.get_device();
    const auto sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    if (!dev.is_gpu() || dev.get_info<sycl::info::device::vendor_id>() != 0x8086 ||
        std::find(sizes.begin(), sizes.end(), size_t(32)) == sizes.end() ||
        std::find(sizes.begin(), sizes.end(), size_t(16)) == sizes.end()) {
        std::printf("gdn_intel_parity: skipped (braucht Intel-GPU mit Sub-groups 16 und 32)\n");
        return 0;
    }
    std::printf("gdn_intel_parity device=%s T=%lld (fp64-Referenz ym, modulo/Strata-Nummerierung)\n",
                dev.get_info<sycl::info::device::name>().c_str(), (long long)T);

    // hbuf wie im Engine-Pfad nach der conv+l2norm-Stufe: q,k eps-l2norm (q OHNE rsqrt(S) - das
    // tragt der Xe-Kern intern, L82, und der Intel-Stage), v roh unver permutation (ref.v-Zeilen
    // zaehlen schon in Strata-Slots, s4-Bench L257); alles in Strata-Dichtlage.
    std::vector<float> hbuf((size_t)T * C);
    for (int64_t t = 0; t < T; ++t) {
        for (int hd = 0; hd < HK; ++hd) {
            float* qrow = &hbuf[(size_t)t * C + (size_t)hd * S];
            float* krow = &hbuf[(size_t)t * C + (size_t)(HK + hd) * S];
            const float* qr = &ref.q[((size_t)t * HK + hd) * S];
            const float* kr = &ref.k[((size_t)t * HK + hd) * S];
            float nq = 0, nk = 0;
            for (int d = 0; d < S; ++d) { qrow[d] = qr[d]; krow[d] = kr[d]; nq += qr[d] * qr[d]; nk += kr[d] * kr[d]; }
            const float iq = 1.f / std::sqrt(nq + NORM_EPS), ik = 1.f / std::sqrt(nk + NORM_EPS);
            for (int d = 0; d < S; ++d) { qrow[d] *= iq; krow[d] *= ik; }
        }
        std::memcpy(&hbuf[(size_t)t * C + (size_t)2 * HK * S], &ref.v[(size_t)t * HV * S], (size_t)HV * S * 4);
    }
    std::vector<float> ssm_a(HV);
    for (int h = 0; h < HV; ++h) ssm_a[h] = -std::exp(ref.alog[h]);   // GGUF-Konvention (engine)

    const size_t n_state = (size_t)S * HV * S, n_y = (size_t)T * HV * S;
    float *d_h = (float*)sycl::malloc_device(C * T * 4, q), *d_ab = (float*)sycl::malloc_device(2 * HV * T * 4, q),
          *d_dt = (float*)sycl::malloc_device(HV * 4, q), *d_ss = (float*)sycl::malloc_device(HV * 4, q),
          *d_gate = (float*)sycl::malloc_device(HV * T * 4, q), *d_beta = (float*)sycl::malloc_device(HV * T * 4, q),
          *d_yS = (float*)sycl::malloc_device(n_y * 4, q), *d_yI = (float*)sycl::malloc_device(n_y * 4, q),
          *d_stS = (float*)sycl::malloc_device(n_state * 4, q), *d_stI = (float*)sycl::malloc_device(n_state * 4, q);
    if (!d_h || !d_ab || !d_dt || !d_ss || !d_gate || !d_beta || !d_yS || !d_yI || !d_stS || !d_stI) {
        std::fprintf(stderr, "malloc fehlgeschlagen\n"); return 2;
    }
    q.memcpy(d_h, hbuf.data(), C * T * 4).wait_and_throw();
    q.memcpy(d_ab, ref.ab.data(), 2 * HV * T * 4).wait_and_throw();
    q.memcpy(d_dt, ref.dt.data(), HV * 4).wait_and_throw();
    q.memcpy(d_ss, ssm_a.data(), HV * 4).wait_and_throw();
    q.memset(d_stS, 0, n_state * 4).wait_and_throw();
    q.memset(d_stI, 0, n_state * 4).wait_and_throw();

    // Arm I zuerst mit stage_only: Stage-Puffer rueckkopieren und gegen die Wirt-Erwartung halten.
    // Das trennt "unsere Stage baut falsche Zahlen" von "der Kern rechnet falsch" (Fenster 14.10.:
    // y max_abs 0,03937 + nicht-finite Werte, S4-Bench mit Host-Prep gleichzeitig PASS).
    strata_intel_gdn::Scratch ig;
    bool stage_ok = true;
    if (do_stagecheck) {
        if (!strata_intel_gdn::run(q, ig, d_h, d_ab, d_dt, d_ss, d_stI, d_yI, T, S, HK, HV, /*stage_only*/ true)) {
            std::fprintf(stderr, "strata_intel_gdn::run(stage_only) fehlgeschlagen (Form? Scratch?)\n"); return 2;
        }
        q.wait_and_throw();
        stage_ok = stagecheck(q, ig, stage_expect(hbuf, ref.ab, ref.dt, ssm_a,
                                                  std::vector<float>(n_state, 0.f), T), T);
    }

    // Arm S: die etablierte Kette (gdn_gates faltet softplus/ssm_a und sigmoid, Xe-Kern normiert nicht mehr).
    strata::prefill::gdn_gates(d_ab, d_dt, d_ss, d_gate, d_beta, T, &q);
    strata::prefill::detail::gdn_xe_launch(q, d_stS, d_h, d_gate, d_beta, d_yS, T).wait_and_throw();

    // Arm I: der Intel-Arm (eigener Scratch, liest dieselben Rohwerte; stage_only-Lauf wurde
    // ueberschrieben — run() baut den Stage vollstaendig neu, ist determiniert).
    if (!strata_intel_gdn::run(q, ig, d_h, d_ab, d_dt, d_ss, d_stI, d_yI, T, S, HK, HV)) {
        std::fprintf(stderr, "strata_intel_gdn::run fehlgeschlagen (Form? Scratch?)\n"); return 2;
    }
    q.wait_and_throw();

    std::vector<float> yS(n_y), yI(n_y), stS(n_state), stI(n_state);
    q.memcpy(yS.data(), d_yS, n_y * 4).wait_and_throw();
    q.memcpy(yI.data(), d_yI, n_y * 4).wait_and_throw();
    q.memcpy(stS.data(), d_stS, n_state * 4).wait_and_throw();
    q.memcpy(stI.data(), d_stI, n_state * 4).wait_and_throw();

    const Stats sYm = compare(yS, ref.ym), iYm = compare(yI, ref.ym), sI = compare(yS, yI);
    double st_maxS = 0, st_maxI = 0, st_d = 0;
    size_t st_nanS = 0, st_nanI = 0;
    for (size_t i = 0; i < n_state; ++i) {
        if (!std::isfinite(stS[i])) ++st_nanS;
        if (!std::isfinite(stI[i])) ++st_nanI;
        st_maxS = std::max(st_maxS, (double)std::fabs(stS[i]));
        st_maxI = std::max(st_maxI, (double)std::fabs(stI[i]));
        st_d = std::max(st_d, (double)std::fabs((double)stS[i] - (double)stI[i]));
    }
    auto nancnt = [](const std::vector<float>& v) {
        size_t n = 0; for (float x : v) if (!std::isfinite(x)) ++n; return n; };
    std::printf("  y   Xe-Arm   vs fp64-ym          : max_abs=%.5f |ref|_mittel=%.4f NaN=%zu\n",
                sYm.max_abs, sYm.scale / n_y, nancnt(yS));
    std::printf("  y   Intel-Arm vs fp64-ym         : max_abs=%.5f |ref|_mittel=%.4f NaN=%zu\n",
                iYm.max_abs, iYm.scale / n_y, nancnt(yI));
    std::printf("  y   Arm vs Arm (Info)            : max_abs=%.5f\n", sI.max_abs);
    // getrennte Maxima: rel~1 mit st_maxI~0 heisst "der eine Arm hat gar nicht geschrieben",
    // rel~1 bei gleichen Maxima heisst "beide schreiben, aber verschiedenes".
    std::printf("  state Xe vs Intel (rel. zum Xe-Max): max_abs=%.6f rel=%.5f |Xe|max=%.6f |Intel|""max=%.6f NaN=%zu/%zu\n",
                st_d, st_maxS > 0 ? st_d / st_maxS : 0.0, st_maxS, st_maxI, st_nanS, st_nanI);

    // Schwellwerte: Diagnoese nach S4-Messbasis, keine Beweiskraft (s4 README/s4_gdn_bench L340-346).
    int bad = 0;
    if (!stage_ok) { std::printf("FAIL: Stage-Puffer weichen von der Wirt-Erwartung ab (siehe stage-Zeilen)\n"); ++bad; }
    if (sYm.nan || iYm.nan || st_nanS || st_nanI) { std::printf("FAIL: Nicht-finite Werte\n"); ++bad; }
    if (sYm.max_abs > 0.02) { std::printf("FAIL: Xe-Arm vs fp64 ueber 0.02 (Erwartung <<)\n"); ++bad; }
    if (iYm.max_abs > 0.02) { std::printf("FAIL: Intel-Arm vs fp64 ueber 0.02 (Fenster 14.10.: 0.03937)\n"); ++bad; }
    if (st_maxS == 0) { std::printf("FAIL: Zustand unveraendert (Arme schrieben nichts?)\n"); ++bad; }
    if (st_maxS > 0 && st_d / st_maxS > 0.15) { std::printf("FAIL: Zustands-Umbau: Arme divergieren rel > 0.15\n"); ++bad; }
    std::printf("gdn_intel_parity %s\n", bad ? "FAILED" : "OK");
    sycl::free(d_h, q); sycl::free(d_ab, q); sycl::free(d_dt, q); sycl::free(d_ss, q); sycl::free(d_gate, q);
    sycl::free(d_beta, q); sycl::free(d_yS, q); sycl::free(d_yI, q); sycl::free(d_stS, q); sycl::free(d_stI, q);
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string ref = std::string(getenv("HOME") ? getenv("HOME") : "") + "/strata/istrata-s4/ref/gdn_ref_T256.bin";
    std::string expect_out;
    bool selftest_only = false, do_stagecheck = true, poison = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--selftest") selftest_only = true;
        else if (a == "--ref") ref = argv[++i];
        else if (a == "--no-stagecheck") do_stagecheck = false;
        else if (a == "--expect") expect_out = argv[++i];
        else if (a == "--poison") poison = true;
        else { std::fprintf(stderr, "usage: gdn_intel_parity [--selftest] [--ref datei] [--expect datei] [--no-stagecheck] [--poison]\n"); return 2; }
    }
    if (poison) {   // A/w/u vor jedem Kernstart mit fp16-NaN: beweist, dass die Nullfuellung entbehrlich ist
        setenv("STRATA_GDN_WS_POISON", "1", 1);
        std::printf("POISON an: A/w/u vor jedem Kernstart mit fp16-NaN gefuellt (rc!=0 = Audit false)\n");
    }
    const int rc_st = selftest();
    if (selftest_only) return rc_st;
    if (rc_st) return rc_st;
    if (!expect_out.empty()) {   // wirtsseitig: Erwartungsdatei der Stage-Puffer schreiben (ohne GPU)
        Ref r;
        if (!r.load(ref)) return 2;
        std::vector<float> hbuf((size_t)r.T * C, 0.f), ssm_a(HV);
        for (int64_t t = 0; t < r.T; ++t) {
            for (int hd = 0; hd < HK; ++hd) {
                float* qrow = &hbuf[(size_t)t * C + (size_t)hd * S];
                float* krow = &hbuf[(size_t)t * C + (size_t)(HK + hd) * S];
                const float* qr = &r.q[((size_t)t * HK + hd) * S];
                const float* kr = &r.k[((size_t)t * HK + hd) * S];
                float nq = 0, nk = 0;
                for (int d = 0; d < S; ++d) { qrow[d] = qr[d]; krow[d] = kr[d]; nq += qr[d] * qr[d]; nk += kr[d] * kr[d]; }
                const float iq = 1.f / std::sqrt(nq + NORM_EPS), ik = 1.f / std::sqrt(nk + NORM_EPS);
                for (int d = 0; d < S; ++d) { qrow[d] *= iq; krow[d] *= ik; }
            }
            std::memcpy(&hbuf[(size_t)t * C + (size_t)2 * HK * S], &r.v[(size_t)t * HV * S], (size_t)HV * S * 4);
        }
        for (int h = 0; h < HV; ++h) ssm_a[h] = -std::exp(r.alog[h]);
        StageExp e = stage_expect(hbuf, r.ab, r.dt, ssm_a, std::vector<float>((size_t)S * HV * S, 0.f), r.T);
        std::printf("Stage-Erwartung T=%lld (TP=%lld, Anfangszustand 0) -> %s\n",
                    (long long)r.T, (long long)e.TP, expect_out.c_str());
        return write_expect(expect_out, e, r.T) ? 0 : 2;
    }
    return gpu_case(ref, do_stagecheck);
}
