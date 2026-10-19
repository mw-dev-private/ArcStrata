# SYCL port layout: which tree is live, which is reference

The Intel SYCL backend (what runs in production on the Arc B70) lives entirely under `sycl/`.
The CUDA originals under `src/` are **never compiled by the SYCL build** - but they are not dead
weight either. Know which is which before editing or deleting anything.

## How the SYCL build resolves sources

`sycl/CMakeLists.txt` has a `strata_resolve()` function: every entry named `src/X.cu` is mapped to
the migrated copy `sycl/src/X.dp.cpp`, and a missing migrated copy is a `FATAL_ERROR`. The `.cu`
names are a manifest, not build inputs. Plain `.cpp` entries (and everything under
`src/core/`, `src/spec/`, `src/platform/`, `src/ngram/`, `src/prefill/intel_gdn/`) are compiled
**directly from `src/`** when `sycl/` carries no copy - edits there affect every backend.

Measured 2026-10-19: `src/` holds 127k lines, of which 49k are CUDA (`.cu`/`.cuh`); the live
migrated kernels are 50k lines of `.dp.cpp` under `sycl/src/`.

## Why the `.cu` files stay

1. **Upstream parity.** Upstream Strata evolves in CUDA. Merging an upstream feature means diffing
   its `.cu` against the original and porting the delta into the `.dp.cpp` - this is exactly how
   the int8 prefill (v0.2.0-beta) was merged on 2026-10-19. Without the originals, every future
   merge becomes a blind hand translation.
2. **The HIP backend compiles the same `.cu` sources** (gfx906 compat layer, see
   [AMD_HIP.md](AMD_HIP.md)). The CUDA tree is dead only to the SYCL build, not to the project.
3. **Parity harnesses** (`sycl/src/kernels/*_parity.cpp`) validate the port against CUDA numerics.

## Rules

- A production SYCL fix goes into `sycl/` (the `.dp.cpp`). Editing a `.cu` never reaches the
  SYCL binary - it only desynchronizes the reference.
- Do not delete the `.cu` files: `strata_resolve()` references them by name, and ~60 CMake
  entries would need rewriting first. There is no runtime or compile-time gain either way -
  they were never compiled.
- The ~1500 `cuda` mentions inside `sycl/src` are DPCT migration comments and log strings, not
  live CUDA calls.
