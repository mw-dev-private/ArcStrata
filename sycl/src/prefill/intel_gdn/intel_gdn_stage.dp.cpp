// intel_gdn_stage.dp.cpp — Engine-Seite des Intel-GDN-Pfads (Schritt 2+3 der Patch-Serie).
// Hier lebt alles, was die Engine torch-frei an Intels vendored kernel (gdn_kernels_xe2.hpp,
// namespace gdn, Zeiger-Schnittstelle kernel_launcher) anhaengt:
//   * Stage-Durchzuege: dichtes hbuf [T,C] fp32 -> polstrierende fp16-Puffer [TP,...] mit
//     V-Kopf-Permutation (h <-> hv'=(h%HK)*rep+h/HK); a/b [T,2HV] fp32 -> [HV,TP] fp32, transponiert,
//     sigmoid(b) gefaltet; dt -> fp16; ssm_a -> A_log'=log(-ssm_a); q zusaetzlich x rsqrt(S), k ohne
//     (Intels l2norm-Semantik, im S4-Bench gegen fp64-Referenz gepreift — die Norm selbst hat schon
//     Stratas Conv-Schritt geliefert, hier laeuft nur die Skalierung mit).
//   * Zustands-Umbau [k][h][v] (Strata/Decode) <-> [hv][v][k] (Intel) um den Kern herum, damit der
//     Decode-Pfad unveraendert liesst.
//   * Ausgabe-Rueckpermutation Intel-Kopf -> Strata-Slot, fp16 -> fp32 y (derselbe Durchzug).
//   * Workspace A/w/u je Aufruf genullt — wie Intels eigene Huelle (torch::zeros), die im S4-Bench
//     mitgemessen wurde; Puffer selbst sind persistent (graph-capture-faehig, nur vor dem Capture
//     waechst ensure()).
#include "intel_gdn.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>

#include "gdn_kernels_xe2.hpp"   // vendored; definiert gdn::kernel_launcher (Header-only, eine TU!)

namespace strata_intel_gdn {

static constexpr int kChunk = 64;         // gdn::chunk_size_xe2
static constexpr int kMaxPad = kChunk - 1;
static constexpr int HV_ = 48;            // formfeste Geometrie (ensure() allokiert fest 48 Koepfe, S=128)

bool supported(int S, int HK, int HV) { return S == 128 && HK == 16 && HV == 48; }

// Kopfpaarung perm_h2i/perm_i2h: in intel_gdn.hpp (damit der Parity-Selftest dieselbe Funktion prüft).

void Scratch::free_all() {
  if (!qhold) return;   // nichts allokiert
  sycl::free(qpad, *qhold); sycl::free(kpad, *qhold); sycl::free(vpad, *qhold);
  sycl::free(ia, *qhold); sycl::free(ib, *qhold); sycl::free(dt16, *qhold); sycl::free(alogp, *qhold);
  sycl::free(iout, *qhold); sycl::free(istate, *qhold);
  sycl::free(A, *qhold); sycl::free(w, *qhold); sycl::free(u, *qhold);
  sycl::free(qsl, *qhold); sycl::free(ci, *qhold); sycl::free(hinit, *qhold); sycl::free(hvoff, *qhold);
  qpad = kpad = vpad = dt16 = iout = A = w = u = nullptr;
  ia = ib = alogp = istate = nullptr; qsl = ci = nullptr; hinit = nullptr; hvoff = nullptr;
  tp_cap = 0;
}

Scratch::~Scratch() {
  free_all();
  delete qhold;
}

bool Scratch::ensure(sycl::queue& q, int64_t T) {
  if (tp_cap >= T) return true;
  const int64_t TP = ((T + kChunk - 1) / kChunk) * kChunk;
  if (!qhold) qhold = new sycl::queue(q);
  free_all();
  qpad = sycl::malloc_device((size_t)TP * 2048 * 2, q);
  kpad = sycl::malloc_device((size_t)TP * 2048 * 2, q);
  vpad = sycl::malloc_device((size_t)TP * 6144 * 2, q);
  ia = (float*)sycl::malloc_device((size_t)48 * TP * 4, q);
  ib = (float*)sycl::malloc_device((size_t)48 * TP * 4, q);
  dt16 = sycl::malloc_device(48 * 2, q);
  alogp = (float*)sycl::malloc_device(48 * 4, q);
  iout = sycl::malloc_device((size_t)T * 48 * 128 * 2, q);
  istate = (float*)sycl::malloc_device((size_t)48 * 128 * 128 * 4, q);
  A = sycl::malloc_device((size_t)48 * (T + kMaxPad) * kChunk * 2, q);
  w = sycl::malloc_device((size_t)48 * (T + kMaxPad) * 128 * 2, q);
  u = sycl::malloc_device((size_t)48 * (T + kMaxPad) * 128 * 2, q);
  qsl = (int*)sycl::malloc_device(2 * sizeof(int), q);
  ci = (int*)sycl::malloc_device(sizeof(int), q);
  hinit = (unsigned char*)sycl::malloc_device(1, q);
  hvoff = (int*)sycl::malloc_device(48 * sizeof(int), q);
  if (!qpad || !kpad || !vpad || !ia || !ib || !dt16 || !alogp || !iout || !istate || !A || !w || !u ||
      !qsl || !ci || !hinit || !hvoff) { tp_cap = 0; return false; }
  // Kopfoffset-Tabelle: Engine-out_norm (kernels.dp.cpp) liesst Intels fp16 iout damit direkt in
  // Strata-Slot-Reihenfolge — spaart den out_write-Durchzug (604 MB/Schicht bei T=16384).
  q.submit([&](sycl::handler& cgh) {
    int* ho = hvoff;   // Kernel darf this nicht implizit erfassen — Zeiger explizit einfangen
    cgh.parallel_for<class IGHeadOff>(sycl::range<1>(48), [ho](sycl::item<1> it) {
      const int slot = (int)it.get_id(0);
      ho[slot] = perm_h2i(slot, 16, 3);
    });
  });
  // qsl/ci/hinit fuellen wir geraeteseitig je run() (IGFillCtrl): ein Host-memcpy aus dem Member hvv
  // waere asynchron auf pageablem Speicher — die naechste Schicht ueberschriebe hvv, bevor die Kopie
  // laeuft (DEVICE_LOST-Verdaechtig 14.10., Auftrag 4a).
  tp_cap = T;
  return true;
}

// --- Stage-Kerne -------------------------------------------------------------
// (Deklaration der Einzel-Durchzuege in intel_gdn.hpp; Definitionen hier, damit alle Geraet-Kerne
//  im vendored Uebersetzungsfeld mitlaufen - siehe Sub-Group-Override im CMake.)
class IGStageQKV;
class IGStageAB;      // fusioniert: a/b + Koepfe + Steuerfelder (Runde 3, Aufgabe 4)
class IGStateIn;
class IGStateOut;
class IGOutWrite;

// hbuf [T,C] fp32 (q|k|v) -> qpad/kpad/vpad fp16 [TP,...]; q mit rsqrt(S), k ohne; t>=T genullt;
// v-Zeilen in Intel-Kopf-Nummerierung.
void stage_qkv(sycl::queue& q, const float* hbuf, void* qpad, void* kpad, void* vpad,
                      int64_t T, int64_t TP, int S, int HK, int HV) {
  const int64_t C = 2 * HK * S + HV * S;
  const int64_t HKS = HK * S, HVS = HV * S;
  const float qsc = 1.0f / std::sqrt((float)S);
  const int rep = HV / HK;
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class IGStageQKV>(sycl::range<3>(1, (size_t)TP, (size_t)C), [=](sycl::item<3> it) {
      const int64_t t = it.get_id(1), c = it.get_id(2);
      const float x = (t < T) ? hbuf[t * C + c] : 0.0f;
      if (c < HKS) {
        ((sycl::half*)qpad)[t * HKS + c] = sycl::half(x * qsc);
      } else if (c < 2 * HKS) {
        ((sycl::half*)kpad)[t * HKS + (c - HKS)] = sycl::half(x);
      } else {
        const int64_t cv = c - 2 * HKS;
        const int slot = (int)(cv / S), d = (int)(cv % S);
        const int hv = perm_h2i(slot, HK, rep);
        ((sycl::half*)vpad)[t * HVS + (int64_t)hv * S + d] = sycl::half(x);
      }
    });
  });
}

// ab [T,2HV] fp32 (a|b) -> ia/ib [HV,TP] fp32, transponiert, sigmoid(b), Intel-Nummerierung, t>=T 0.
// Fusion (Runde 3, Aufgabe 4): dieselbe Gitterspitze haengt die Koepfe (dt16/alogp, hv<HV) und die
// Steuerfelder (qsl/ci/hinit, letzter Index TP) an — statt drei Starts einer wird's einer. Die
// Koepfe-Arbeit war ein eigener 48-Item-Kern (IGStageHeads), die Steuerfelder ein single-item-Kern
// (IGFillCtrl); beide sind gegen den a/b-Durchzug rauschfrei, Reihenfolge innerhalb des Items egal.
void stage_gates(sycl::queue& q, const float* ab, float* ia, float* ib, const float* dt,
                 const float* ssm_a, void* dt16, float* alogp, int* qsl, int* ci,
                 unsigned char* hinit, int64_t T, int64_t TP, int HK, int HV) {
  const int rep = HV / HK;
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class IGStageAB>(sycl::range<3>(1, (size_t)TP + 1, (size_t)HV + 1),
                                      [=](sycl::item<3> it) {
      const int64_t t = it.get_id(1); const int hv = (int)it.get_id(2);
      if (t == TP) {
        if (hv == HV) { qsl[0] = 0; qsl[1] = (int)T; ci[0] = 0; hinit[0] = 1; }   // Steuerfelder
        else {
          const int h = perm_i2h(hv, HK, rep);
          ((sycl::half*)dt16)[hv] = sycl::half(dt[h]);
          // ssm_a = -exp(A_log) der Rolle; umgekehrt A_log = log(-ssm_a). ssm_a >= 0 (kaputte Rolle) ->
          // -1e30, damit der Kern dort ohne Zustand weiterrechnet und der Parity-Test es findet.
          alogp[hv] = (ssm_a[h] < 0.0f) ? std::log(-ssm_a[h]) : -1.0e30f;
        }
        return;
      }
      const int h = perm_i2h(hv, HK, rep);
      float a = 0.0f, b = 0.0f;
      if (hv == HV) return;   // (t<TP, hv==HV): nichts zu tun — Gitterfuellung, kein Speicherzugriff
      if (t < T) { a = ab[t * 2 * HV + h]; b = ab[t * 2 * HV + HV + h]; }
      ia[(int64_t)hv * TP + t] = a;
      // Polsterzeilen wie im s4-Bench genullt (ib nicht sigmoid(0)=0.5); Intels Kern liesst t>=seq_len
      // zwar nie (chunk-Grenzen, gdn_kernels_xe2.hpp L984/L997-1010), aber gleiche Eingabe wie die Messbasis.
      ib[(int64_t)hv * TP + t] = (t < T) ? 1.0f / (1.0f + std::exp(-b)) : 0.0f;
    });
  });
}

// Strata [k][h][v] -> Intel [hv][v][k] (Anfangszustand hinein).
void state_in(sycl::queue& q, const float* state, float* istate, int S, int HK, int HV) {
  const int rep = HV / HK;
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class IGStateIn>(sycl::range<3>(1, (size_t)HV, (size_t)S * S), [=](sycl::item<3> it) {
      const int hv = (int)it.get_id(1); const int vk = (int)it.get_id(2);
      const int v = vk / S, k = vk % S;
      const int h = perm_i2h(hv, HK, rep);
      istate[(int64_t)hv * S * S + vk] = state[(int64_t)k * HV * S + h * S + v];
    });
  });
}

// Intel [hv][v][k] -> Strata [k][h][v] (Endzustand zurueck, Decode-pfad-treu).
void state_out(sycl::queue& q, const float* istate, float* state, int S, int HK, int HV) {
  const int rep = HV / HK;
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class IGStateOut>(sycl::range<3>(1, (size_t)HV, (size_t)S * S), [=](sycl::item<3> it) {
      const int hv = (int)it.get_id(1); const int vk = (int)it.get_id(2);
      const int v = vk / S, k = vk % S;
      const int h = perm_i2h(hv, HK, rep);
      state[(int64_t)k * HV * S + h * S + v] = istate[(int64_t)hv * S * S + vk];
    });
  });
}

// iout fp16 [T,HV,S] Intel-Nummerierung -> y fp32 [T,HV*S] Strata-Slots.
void out_write(sycl::queue& q, const void* iout, float* y, int64_t T, int S, int HK, int HV) {
  const int rep = HV / HK;
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class IGOutWrite>(sycl::range<3>(1, (size_t)T * HV, (size_t)S), [=](sycl::item<3> it) {
      const int64_t th = it.get_id(1); const int d = (int)it.get_id(2);
      const int64_t t = th / HV; const int h = (int)(th % HV);
      const int hv = perm_h2i(h, HK, rep);
      y[th * S + d] = (float)((const sycl::half*)iout)[(t * HV + hv) * S + d];
    });
  });
}

// Workspace nullen — SEIT Runde 3 optional (memset-Audit, Quelle gdn_kernels_xe2.hpp):
//  * A: compute_A_kernel schreibt JEDES 64x64-Zell vor Inverse (L257 clear, L258 K-K^T,
//    L273-278 Diag=1/obeere Dreieck explicit 0, L283 ganzes Tile unprädikativ zurück).
//  * u: chunk_compute_wu L837-848 ganzes 64x128-Tile je Kopf/Chunk, vor FwdO-Lese L1021.
//  * w: dito L874-885, aber durch Guard L850 (chunk!=0 || has_initial_state) — Engine uebergibt
//    immer hinit=1 (stage_gates), also vollstaendig; mit hinit=false waere chunk0-w Pflicht-Null.
//  * Zeilen [TP, T+63): kein Kernel indexiert sie (chunk-Zahl = ceil(seq_len/64), Schrittweite
//    total_virtual_seqlen=TP); Intels torch::zeros (Original L1560-1568) ist Zuteilungs-Artefakt.
// Pflicht bleibt das Nullen NUR bei hinit=false. Fenster-A/B-Fallback: STRATA_GDN_WS_MEMSET=1.
void memset_ws(sycl::queue& q, Scratch& sc, int64_t T) {
  q.memset(sc.A, 0, (size_t)HV_ * (T + kMaxPad) * kChunk * 2);
  q.memset(sc.w, 0, (size_t)HV_ * (T + kMaxPad) * 128 * 2);
  q.memset(sc.u, 0, (size_t)HV_ * (T + kMaxPad) * 128 * 2);
}

// Poison (Nachlese Besitzer-Fenster 14.10.): A/w/u vor dem Kernstart mit fp16-NaN fuellen statt mit
// Nullen. Beweist die memset-Entbehrlichkeit, die das normale Parity-Rennen nicht beweist: frisch
// zugeteilter Speicher kann zufaellig Nullen enthalten, und die absolute Endlichkeitsprobe im Bench
// trennt nicht zwischen "synthetische Gates detonieren" und "Kern liesst Ungeschriebenes". Laest der
// Kern je auf ungeschriebenen A/w/u-Zellen, schlaegt NaN in iout/y/istate durch -> Paritaet faellt.
// Aktivierung: Umgebungsvariable STRATA_GDN_WS_POISON=1 (intel_gdn_parity --poison setzt sie) oder
// direkter Aufruf (bench/gdn_stage_bench.cpp, y1-vs-y2-Vergleich bei zwei verschiedenen Startfuellungen).
void poison_ws(sycl::queue& q, Scratch& sc, int64_t T) {
  const size_t na = (size_t)HV_ * (T + kMaxPad) * kChunk, nwu = (size_t)HV_ * (T + kMaxPad) * 128;
  uint16_t* pa = reinterpret_cast<uint16_t*>(sc.A);
  uint16_t* pw = reinterpret_cast<uint16_t*>(sc.w);
  uint16_t* pu = reinterpret_cast<uint16_t*>(sc.u);
  q.parallel_for<class IGPoisonWS>(sycl::range<1>(na + 2 * nwu), [=](sycl::item<1> it) {
    const uint16_t qnan = 0x7E00;  // fp16 quiet NaN: Exponent alles-1, Mantissa != 0
    const size_t i = it.get_id(0);
    if (i < na) pa[i] = qnan;
    else if (i < na + nwu) pw[i - na] = qnan;
    else pu[i - na - nwu] = qnan;
  });
}

// Nur der vendored Kern (fp16 x fp32-Zustand: die einzige instantiierte Kombination, Messbasis des Fensters).
void kernel_only(sycl::queue& q, Scratch& sc, int64_t T, int S, int HK, int HV) {
  const int64_t TP = ((T + kChunk - 1) / kChunk) * kChunk;
  gdn::kernel_launcher<cute::half_t, float>(
      q,
      reinterpret_cast<cute::half_t*>(sc.iout),
      reinterpret_cast<const cute::half_t*>(sc.qpad),
      reinterpret_cast<const cute::half_t*>(sc.kpad),
      reinterpret_cast<const cute::half_t*>(sc.vpad),
      reinterpret_cast<cute::half_t*>(sc.A),
      reinterpret_cast<cute::half_t*>(sc.w),
      reinterpret_cast<cute::half_t*>(sc.u),
      sc.ib, sc.ia, sc.alogp,
      reinterpret_cast<const cute::half_t*>(sc.dt16),
      sc.istate, HV * S * S,
      sc.qsl, sc.ci, reinterpret_cast<const bool*>(sc.hinit),
      /*token_indx*/ nullptr,
      /*batch_size*/ 1, /*total_virtual_seqlen*/ (int)TP,
      HK, S, HV, S);
}

bool run(sycl::queue& q, Scratch& sc, const float* hbuf, const float* ab, const float* dt,
         const float* ssm_a, float* state, float* y, int64_t T, int S, int HK, int HV,
         bool stage_only, bool skip_out) {
  if (!supported(S, HK, HV) || T <= 0) return false;
  if (!sc.ensure(q, T)) return false;
  const int64_t TP = ((T + kChunk - 1) / kChunk) * kChunk;

  stage_qkv(q, hbuf, sc.qpad, sc.kpad, sc.vpad, T, TP, S, HK, HV);
  stage_gates(q, ab, sc.ia, sc.ib, dt, ssm_a, sc.dt16, sc.alogp, sc.qsl, sc.ci, sc.hinit, T, TP, HK, HV);
  state_in(q, state, sc.istate, S, HK, HV);
  static const bool ws_zero = [] {
    const char* e = std::getenv("STRATA_GDN_WS_MEMSET");
    return e && e[0] == '1';   // A/B-Fallback unters Fenster; Pflicht nur bei hinit=false (Audit L850)
  }();
  static const bool ws_poison = [] {
    const char* e = std::getenv("STRATA_GDN_WS_POISON");
    return e && e[0] == '1';   // Beweis-Modus: NaN statt 0 — erreicht NaN die Ausgabe, ist das Audit false
  }();
  if (ws_poison) poison_ws(q, sc, T);
  else if (ws_zero) memset_ws(q, sc, T);
  if (stage_only) return true;   // Paritaets-Rueckkopie: Stage-Puffer stehen, der Kern laeuft noch nicht

  kernel_only(q, sc, T, S, HK, HV);
  if (!skip_out) out_write(q, sc.iout, y, T, S, HK, HV);   // skip_out: Engine-out_norm liesst iout fp16 direkt (Aufgabe 5)
  state_out(q, sc.istate, state, S, HK, HV);
  return true;
}

}  // namespace strata_intel_gdn
