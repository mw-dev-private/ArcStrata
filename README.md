# Strata + Qwen3.8-Flash-Next on an Intel Arc Pro B70 32 GB — my fork

*mw-dev-private, October 2026.* A fork of [Strata](https://github.com/Niko1221/Strata) (MIT) with the SYCL work and the
settings I run every day to serve **Qwen3.8-Flash-Next (GSQ-RCO IQ3_S)** on an **Intel Arc Pro B70** behind a coding
agent, a local assistant and an MCP client. Work done with several coding agents (Claude/Codex/OpenCode) as the
assistants; the direction, the hypotheses and the dead ends are mine.

**All the numbers on this page are mine, measured on this PC.** For Strata itself, its install and its own speed
tables see the [upstream README](https://github.com/Niko1221/Strata#readme) and, for the Intel port in general,
[docs/INTEL.md](docs/INTEL.md). The exact live state of today — binary, commit, flags, deployment chain, open
findings — is snapshotted in [docs/PROD-STATE-2026-10-20.md](docs/PROD-STATE-2026-10-20.md).

> [!IMPORTANT]
> **This fork is Intel Arc only, and it is only the B70.** Everything here is the Intel engine: the SYCL sources under
> `sycl/`, built with `-DSTRATA_ENABLE_SYCL=ON`, after which upstream's own CMake configures nothing else and stops
> the build outright if `STRATA_ENABLE_CUDA` or `STRATA_ENABLE_HIP` is set as well — the backends are mutually
> exclusive by design. So **no CUDA, no ROCm, no Metal/Mac, no multi-GPU, no Windows**: those paths are upstream's,
> they are untouched by this branch, never compiled by my build command, never run and never measured here.
> Within Intel: every change was written, compiled, tuned and measured on one card, the **Arc Pro B70 32 GB**
> (Battlemage, PCI device `8086:e223`, `xe` kernel driver). The box has a second Arc, an **Arc Pro B60**
> (`8086:e211`), which I use as a test card for kernel experiments and for an embedding backend — it is not the
> target of this branch and none of the numbers below were measured on it. Nothing is validated on an A-series Arc,
> an iGPU or any other Intel card. If your card is not a B70, treat this branch as untested and use upstream.

> [!NOTE]
> **Every speed number here is measured the way I use the server every day**: 262,144-token context, thinking off,
> temperature 0, batch 3 slots with MTP, Strata's own bench tool (`python tools/strata_bench.py --runs 1`).
> Upstream's published tables use 4K answers on 32K prompts. Context and batching cost speed, and this machine's
> profile is not the profile of a 16 GB card.

## My PC

| | |
| --- | --- |
| **GPU (production)** | Intel Arc Pro B70, 32 GB, PCI `04:00.0`, device `8086:e223`, Battlemage, `xe` driver (`/dev/dri/renderD128`), Level Zero compute runtime 26.35.39758.10 |
| **GPU (test card)** | Intel Arc Pro B60, PCI `0b:00.0`, device `8086:e211` (`/dev/dri/renderD129`) — the card the kernel experiments ran on, and the card my embedding backend (`bge-m3`) sits on. **No number on this page comes from it** |
| **CPU / RAM** | Intel Core i7-14700F, 123 GiB DDR4-3200 on an MSI B760 Gaming Plus WiFi DDR4 board |
| **System** | Linux, kernel 7.0.0-34-generic |
| **Compiler** | Intel oneAPI DPC++/C++ 2026.1.1 — `icx` and `icpx` both set explicitly |
| **Build gate** | `-DSTRATA_ENABLE_SYCL=ON`: only `sycl/` is configured; the CUDA/HIP combination is a CMake error |
| **Device code** | SPIR-V, JIT-compiled for the B70 at first start with a persistent cache (`SYCL_CACHE_PERSISTENT=1`, fixed `SYCL_CACHE_DIR`), device picked by `ONEAPI_DEVICE_SELECTOR=level_zero:0`. `STRATA_SYCL_AOT` is deliberately **unset**; AOT measured ±0 (see the archive branches) |
| **Model** | Qwen3.8-Flash-Next GSQ-RCO **IQ3_S** (two GGUF shards, the second passed as `--ple-gguf`) |
| **Engine** | upstream **v0.2.0-beta** base (`v0.2.0-beta` tag) plus my int8/batch/slot-copy line — running binary `strata-v0145-slotcopy-diagnostics-1d77cd2d-c8d97081`, source on branch **`engine-int8-v020-instr`** |
| **Serve / control** | this branch: upstream v0.1.45 base (`477ac13`) + port + tokenizer optimizers; config in [prod-config/PT3M-262k.sample.json](prod-config/PT3M-262k.sample.json) |
| **Runtime settings** | `--max-context 262144`, `--kv int8`, `--batch 3 --spec 4` (MTP), `--pool-workers 12`, `--pcie-frac 0.20`, `--expert-cache auto`, `--conversation-cache-mib 16384 --conversation-cache-slots 8 --kv-resident 20480`, `--stream-experts`, vision on; armed gates: `STRATA_NATIVE_GRAPH=1`, `STRATA_BATCH_MTP=1`, `STRATA_TAIL_SKIP_IN_BATCH=1`, `STRATA_PREFILL_CPU_SHARE_BATCH=1`, `STRATA_SLOT_COPY_DIRECT=1`, int8 DPAS prompt experts **on by default** on Battlemage |

## How fast it runs here

Measured 2026-10-20 against the live server (12 pool workers, frac 0.20, batch 3, single-stream suite, `--runs 1`,
median of the suite's own repetitions):

| what | measured | notes |
| --- | --- | --- |
| **writes answers**, short chat | **67.7 tok/s** (60.7–80.6, n=3) | |
| **an agent session** | **62.4 tok/s** (53.7–74.1, n=5) | time to first token 2.99 s |
| **reads a 4K prompt** | **1,423 tok/s** | |
| **reads a 16K prompt** | **1,803 tok/s** | |

The older 131K-era tables, the A/B port charts and the full flag reference (`B70.md`) live on the archive branches
`v0145-b70-sycl-fast` and `docs-b70` — same upstream base (`477ac13`), kept untouched.

One honesty note that matters for reproductions: with `STRATA_TAIL_SKIP_IN_BATCH=1` a slot's answer **depends on
which other slots are busy** (the skip decision sees the whole window), and the serve path is not bit-stable
run-to-run on long greedy generations. Parity tests need an identical side load, not only an identical config.

## What changed since the first version of this README

- **The engine moved to upstream v0.2.0-beta** and carries: the int8 DPAS expert prefill (on by default on
  Battlemage, `STRATA_PREFILL_MMQ=0` turns it off), conversation-cache **slot KV copies** on device
  (`STRATA_SLOT_COPY_DIRECT=1`), **batch prefill CPU share** with an exclusive pool guard
  (`STRATA_PREFILL_CPU_SHARE_BATCH=1`), **tail routing skip inside batch windows**
  (`STRATA_TAIL_SKIP_IN_BATCH=1`), batch MTP, verify instrumentation (stamps, fetch counters) and the 262K context.
- **Tokenizer (Python side)**: two measured optimizers — byte mapping through a prebuilt `str.translate` table
  (mapping 52–68 % faster, ~1–2 % cold end to end) and merge rank keys deduped onto the vocabulary's own strings
  (**−28.7 MB RSS at load**, measured in fresh processes). A cache-refresh experiment exists as a knob but is **off
  by default**: bounded FIFO refresh helps cold-diverse workloads (100 → 1 BPE calls) yet hurts resending
  conversations 6× — details in `docs/PROD-STATE-2026-10-20.md`.
- **Context doubled to 262,144 tokens**, batch serve with 3 slots is the normal mode, pool workers 12 at PCIe frac
  0.20 is today's baseline.

## What is in this fork

- **[docs/PROD-STATE-2026-10-20.md](docs/PROD-STATE-2026-10-20.md)** — the live state as a dated snapshot: binary,
  source commits, config, deployment/rollback chain, measured numbers, open findings from the code review.
- **[prod-config/PT3M-262k.sample.json](prod-config/PT3M-262k.sample.json)** — the config the server actually runs
  (MCP bearer token redacted).
- **[PORT-0145.md](PORT-0145.md)** — the v0.1.45 production-port note from 17 October.
- **[docs/INTEL.md](docs/INTEL.md)** and **[docs/INTEL_ARC.md](docs/INTEL_ARC.md)** — the Intel port logs: what was
  tried, measured and dropped (FEK fusion, XMX prefill GEMM, AOT kernels, ESIMD, W4A16, the int8 MMQ chain and
  more — all with the measurements that ended them).
- [sycl/README_q6_planes.md](sycl/README_q6_planes.md), [sycl/README_prefill_refill.md](sycl/README_prefill_refill.md)
  — in-place Q6 head planes, asynchronous prefill loan refill.

Most engine changes are **opt-in by environment variable**; the int8 prompt experts are the one default-on change.
**Branches:** `main` = the serve/control line that the running server imports (this tree);
`engine-int8-v020-instr` = the engine source line of the running binary (v0.2.0-beta + my changes, tip `1d77cd2d`);
`v0145-b70-sycl-fast` and `docs-b70` = the archived v0.1.45-era state with the flag reference and charts. Nothing
flows back upstream — this is my production box.

Speeds depend on the card, the RAM and the kind of text: these are one PC's numbers, not a promise. Nothing here was
tested for multi-GPU or vision throughput.

License: MIT, as upstream.
