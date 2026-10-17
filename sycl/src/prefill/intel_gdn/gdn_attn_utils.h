// Vendored copy of gdn_attn_utils.h
// Quelle: ~/src/vllm-xpu-kernels, vllm_xpu_kernels 0.1.14.1 (Checkout ohne .git, Datei-Datum 2026-08-27).
// Lizenz: siehe Copyright-Kopf der Datei (BSD-3-Clause, Intel Corporation) bzw. Repo-LICENSE Apache-2.0.
// cutlass-sycl 4.2.1 (NVIDIA, BSD-3): ~/src/vllm-xpu-kernels/.deps/cutlass-sycl-src,
// pinned auf CUTLASS_REVISION 87f6850680a580654b9ea2c80dbc01aeb36ad231 (upstream-CMakeLists L325).
// Stand: 2026-10-14, uebernommen fuer STRATA_GDN_INTEL (siehe README_gdn.md).

#pragma once

namespace gdn {

static constexpr float l2norm_eps = 0.000001f;

#ifdef VLLM_XPU_ENABLE_XE2
static constexpr int chunk_size_xe2 = 64;
#endif

enum class ActMode {
  silu = 0,
  swish = 1,
};

}  // namespace gdn