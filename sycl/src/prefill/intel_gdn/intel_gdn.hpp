#pragma once
// Duenne torch-freie Schnittstelle zu Intels chunk_gated_delta_rule_xe2 (vendored, siehe
// gdn_kernels_xe2.hpp). Nur gebaut mit STRATA_INTEL_GDN; zur Laufzeit opt-in ueber STRATA_GDN_INTEL=1.
// Protocol (im S4-Bench paritaetsgeprueft, siehe ~/strata/istrata-s4): batch=1, T_Polster=ceil(T/64)*64,
// Eingangsdaten vom Stage-Durchzug in Intel-Nummerierung (V-Kopf-Permutation h <-> hv'=(h%HK)*rep+h/HK)
// und Intel-Zustandslayout [hv][v][k]; Ausgabe zurueck in Strata-Slot-Nummerierung und [k][h][v].
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

namespace strata_intel_gdn {

// Unterstuetzte Form (Bench-Messbasis; alles andere faellt auf gdn_xe_launch zurueck).
bool supported(int S, int HK, int HV);

// Kopfpaarung (wie s4_gdn_bench.cpp): Strata-Slot h <-> Intel-Kopf hv'. Steht hier, damit die
// Stage-Durchzuege und der Wirtsselftest von gdn_intel_parity.cpp dieselbe Funktion verwenden.
inline int perm_h2i(int h, int HK, int rep) { return (h % HK) * rep + h / HK; }
inline int perm_i2h(int hv, int HK, int rep) { return hv / rep + HK * (hv % rep); }

struct Scratch {
  // alles persistent, waechst bei Bedarf; NULL bis ensure() laeuft
  void* qpad = nullptr;   // fp16 [TP, HK, S]  (q bereits x rsqrt(S))
  void* kpad = nullptr;   // fp16 [TP, HK, S]
  void* vpad = nullptr;   // fp16 [TP, HV, S]  in Intel-Kopf-Nummerierung
  float* ia = nullptr;    // [HV, TP] raw a (Intel cumsummt in-place, wird je Aufruf neu gebaut)
  float* ib = nullptr;    // [HV, TP] sigmoid(b)
  void* dt16 = nullptr;   // fp16 [HV] Intel-Nummerierung
  float* alogp = nullptr; // fp32 [HV] log(-ssm_a), Intel-Nummerierung (Kern rechnet -exp(A_log))
  void* iout = nullptr;   // fp16 [T, HV, S] Intel-Nummerierung
  float* istate = nullptr;// [HV, S, S] Intel-Layout [hv][v][k]
  void* A = nullptr; void* w = nullptr; void* u = nullptr;  // Workspace, je Aufruf genullt
  int* qsl = nullptr;     // {0, T}
  int* ci = nullptr;      // {0}
  unsigned char* hinit = nullptr;  // {1}: Engine liest den Anfangszustand immer (wie gdn_xe)
  int* hvoff = nullptr;   // [HV]: Intel-Kopf je Strata-Slot (perm_h2i) — Engine-out_norm liest iout damit direkt (Runde 3, Aufgabe 5)
  // kein hvv-Hostfeld mehr: qsl/ci/hinit fuellt der Fill-Kern in intel_gdn_stage.dp.cpp
  // (asynchrones memcpy aus pageablem Member-Speicher war der DEVICE_LOST-Verdacht 14.10.)
  int64_t tp_cap = 0;     // kapazitaetsmaessig gedecktes T
  sycl::queue* qhold = nullptr;  // Kopie der Queue von ensure() (sycl::free braucht sie, newer DPC++)
  ~Scratch();
  void free_all();
  bool ensure(sycl::queue& q, int64_t T);
};

// Ein Prefill-GDN-Delta-Schritt. liest: hbuf [T,C] fp32 (conv+l2norm-fertig, q|k|v), ab [T,2HV] fp32
// (raw a, raw b), dt [HV], ssm_a [HV] (erwartet ssm_a<0); state [S,HV,S] fp32 im Strata-Layout
// [k][h][v] (Anfangszustand rein, Endzustand raus, in-place ueber Intel-Umbau); schreibt y [T,HV*S]
// fp32 in Strata-Slot-Nummerierung. true = gelaufen, false = nicht unterstuetzte Form/Puffer.
// Vor dem ersten Aufruf (und vor Graph-Capture) einmal ensure(T) ausserhalb des Captures rufen.
bool run(sycl::queue& q, Scratch& sc, const float* hbuf, const float* ab, const float* dt,
         const float* ssm_a, float* state, float* y, int64_t T, int S, int HK, int HV,
         bool stage_only = false, bool skip_out = false);

// Einzelfunktionen von run(), exponiert fuer bench/gdn_stage_bench.cpp (Zeit je Stufe) und Parity.
// Alle erklaren keine Queue-Synchronisation — in-order-Queue, Timing ueber wait(). Definitionen in
// intel_gdn_stage.dp.cpp (die Geraet-Kerne muessen in der vendored TU bleiben: 16-Lane-Override).
// stage_gates fusioniert stage_ab + Koepfe (dt/A_log) + qsl/ci/hinit in EINEN Durchgang.
void stage_qkv(sycl::queue& q, const float* hbuf, void* qpad, void* kpad, void* vpad,
               int64_t T, int64_t TP, int S, int HK, int HV);
void stage_gates(sycl::queue& q, const float* ab, float* ia, float* ib, const float* dt,
                 const float* ssm_a, void* dt16, float* alogp, int* qsl, int* ci,
                 unsigned char* hinit, int64_t T, int64_t TP, int HK, int HV);
void state_in(sycl::queue& q, const float* state, float* istate, int S, int HK, int HV);
void state_out(sycl::queue& q, const float* istate, float* state, int S, int HK, int HV);
void out_write(sycl::queue& q, const void* iout, float* y, int64_t T, int S, int HK, int HV);
void memset_ws(sycl::queue& q, Scratch& sc, int64_t T);   // A/w/u nullen (memset-Audit, Runde 3)
void poison_ws(sycl::queue& q, Scratch& sc, int64_t T);   // A/w/u mit fp16-NaN fuellen (Beweis-Modus, Nachlese 14.10.)
void kernel_only(sycl::queue& q, Scratch& sc, int64_t T, int S, int HK, int HV);  // nur kernel_launcher

}  // namespace strata_intel_gdn
