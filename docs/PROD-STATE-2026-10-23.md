# Production state — 2026-10-23 (dated snapshot)

What is actually running on this box as of 2026-10-23. Supersedes
[PROD-STATE-2026-10-20.md](PROD-STATE-2026-10-20.md), which is kept as history.

## Live pieces

| piece | state |
| --- | --- |
| engine binary | `prod/bin/strata-v0145-ecap16`, sha256 `3110fd2c4fa0d90c64e1e018947f20a82740d89174a6da296a67902cdc07c97b`, deployed 2026-10-23 12:00 (`prod/deployments/ecap16-20261023-120026/`) |
| engine source | `release/ecap16` = `9aa8d0df` + `1bac83e2` (MAXT guard, upstream #3805) + `dde551ce` (STRATA_VERIFY_ECAP, upstream #3545); the two fixes are in `main` as `70a56976` / `5aab0988` |
| serve / control | live tree `/home/markus/strata/ports/v0.1.45-fast` at `9aa8d0df` (plus the running web files); `main` `eaa2a63a` mirrors the serve fixes (`a179967c`) and the web/docs. The process started 15:32; the repo sync did not restart it. |
| config | live `prod/batch-test/PT3M-262k.json`; sample `prod-config/PT3M-262k.sample.json` (token redacted) |
| args (live) | `--pack iq3_s --native …IQ3_S-00001… --ple-gguf …IQ3_S-00002… --expert-cache auto --stream-experts --spec 3 --spec-min-p 0.5 --max-context 262144 --kv int8 --prefill auto:32768 --vision --vram-reserve-mib 700 --pcie-frac 0.20 --conversation-cache-mib 16384 --conversation-cache-slots 8 --kv-resident 20480 --pool-workers 12 --batch 3` |
| armed gates | `STRATA_VERIFY_ECAP=16`, `STRATA_NATIVE_GRAPH=1`, `STRATA_BATCH_MTP=1`, `STRATA_TAIL_SKIP_IN_BATCH=1`, `STRATA_PREFILL_CPU_SHARE_BATCH=1`, `STRATA_SLOT_COPY_DIRECT=1`, int8 DPAS prompt experts on by default (`STRATA_PREFILL_MMQ=0` disables); fetch counters and trace knobs off |
| health | `/health`: status ok, `loaded: true`, model `Qwen3.8-Flash`, images on; `/v1/status`: engine `0.1.45` |

## Deploy chain

1. 2026-10-23 12:00 — `ecap16` (`prod/deployments/ecap16-20261023-120026/`): binary swap + `STRATA_VERIFY_ECAP=16`.
   Verified at 32K / 400 tokens: 93.8 / 94.2 tok/s (baseline 85.3–85.9), output sha `da080cc5…` identical, prefill
   1758 tok/s, expert hit 99.9 %. Rollback in the deploy folder.
2. 2026-10-23 — the running Python serve fixes (A16/A17, A03f/g/h, A11c, A12b/c/d, frontend, A04a) were already in
   the live tree; they are mirrored to `main` `a179967c`. No engine restart.
3. 2026-10-23 — repo lean cleanup `b4c4258e`: native CUDA, Metal, Windows/ROCm tooling/tests and every benchmark
   result not measured on this PC removed; only `main`, no tags. Repo only: the live build tree still has the
   unused CUDA/Metal files.
4. 2026-10-23 — IQ3_S vs IQ3_XXS bench, both on `ecap16`, one pack at a time, live restored to IQ3_S.

## Tokenizer (Python side, unchanged from 2026-10-20)

The running tree includes the two optimizers measured on 2026-10-20: byte mapping through a prebuilt
`str.translate` table (mapping 52–68 % faster, ~1–2 % cold end to end) and merge-rank-key dedup onto the
vocabulary's own strings (−28.7 MB RSS at load, fresh processes). The cache-refresh knob exists but is off by
default (bounded FIFO helps cold-diverse workloads, hurts resending 6×).

## Measured 2026-10-23 (IQ3_S / IQ3_XXS)

Official Strata bench, one pack at a time on the same server: `strata-v0145-ecap16` + `STRATA_VERIFY_ECAP=16`,
spec 3, parallel 3, `tools/strata_bench.py --runs 3 --clients 4`.

| test | IQ3_S | IQ3_XXS |
| --- | ---: | ---: |
| Short chat (decode) | 60.6 (55.9–77.4) | **64.7** (59.1–83.0) |
| 4K prompt fresh | 1,295.2 (1,256.9–1,301.7) | **1,388.4** (1,341.9–1,410.6) |
| 16K prompt fresh | **1,751.1** (1,737.6–1,760.2) | 1,745.2 (1,731.0–1,803.1) |
| Agent session (decode) | **67.3** (32.1–80.9) | 62.8 (31.9–79.2) |
| 4 clients total | 56.2 (48.2–58.7) | **57.4** (53.3–59.8) |
| TTFT p50 (4 clients) | 2.099 s | **1.882 s** |
| Errors | 0 | 0 |

All differences are inside the 24 % run-to-run spread. The XXS run used the IQ3_S expert profile as its starting
profile (no own learned profile yet), so this is not a clean pack-to-pack comparison.

## Python fixes in the running tree

Mirrored to `main` `a179967c`; tests in `serve/`:

- A16/A17 process ownership and lifecycle; A03f BT token grammar (503, not 400 or a fabricated token);
  A03g/A03h PP/YIELDED/DONE parsing; A11c Intel wrapper pump slot routing; A12b request accounting;
  A12c telemetry; A12d Xe/gpustat parsers; the frontend review fixes; A04a embedding-file error.

## Open findings

- **Carried from 2026-10-20, not re-checked today:** the pool-guard tripwire is latent (a collision would end the
  engine process); the direct-slot-copy parity test is not in CI; `STRATA_TAIL_SKIP_IN_BATCH=1` keeps answers
  side-load dependent and long greedy generations are not bit-stable. Details in the 2026-10-20 snapshot.
- **IQ3_XXS has no own expert profile.** It started from the IQ3_S profile; a fair comparison needs a learned
  XXS profile.
- **Cutover still open.** Python is the serving layer; the Rust server (`server-rs`) is WIP on `rust/phase1`, not
  on `main`, not deployed. Next up: remove Python.
- **The live build tree is not lean.** `/home/markus/strata/ports/v0.1.45-fast` still contains the unused
  CUDA/Metal files; the repository is leaner and was not rebuilt or deployed from the lean tree.
