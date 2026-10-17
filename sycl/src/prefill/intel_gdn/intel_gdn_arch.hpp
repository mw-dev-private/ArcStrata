#pragma once
// torch-freier Ersatz fuer ~/src/vllm-xpu-kernels csrc/utils.h (die GDN-Kerne brauchten daraus nur
// vllm::xpu::is_bmg()). Frage dieselbe SYCL-Geraetearchitektur wie das Original, aber auf der
// Engine-Queue statt ueber torch/c10. Ergebnis wird beim ersten Aufruf gecacht.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>

inline bool intel_gdn_is_bmg(sycl::queue& q) {
  static int cached = -1;
  if (cached < 0) {
    namespace syclex = sycl::ext::oneapi::experimental;
    const auto a = q.get_device().get_info<syclex::info::device::architecture>();
    cached = (a == syclex::architecture::intel_gpu_bmg_g21 ||
              a == syclex::architecture::intel_gpu_bmg_g31) ? 1 : 0;
  }
  return cached == 1;
}
