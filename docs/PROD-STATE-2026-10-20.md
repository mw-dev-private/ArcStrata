# Production state — 2026-10-20 (dated snapshot)

> **Historical snapshot.** This is the 2026-10-20 state. The current state is
> [PROD-STATE-2026-10-23.md](PROD-STATE-2026-10-23.md).

What was running on this box on 2026-10-20. Kept as a dated snapshot; later changes get later snapshots.

## Live pieces

| piece | state |
| --- | --- |
| engine binary | `prod/bin/strata-v0145-slotcopy-diagnostics-1d77cd2d-c8d97081` (sha256 `c8d97081…`) |
| engine source | branch `engine-int8-v020-instr`, tip `1d77cd2d`: upstream **v0.2.0-beta** (`9f8cb6ed` merge) + slot-KV copies + batch-prefill CPU share + TailInBatch + profilefix + verify instrumentation (`2fdcc351..615a4946`) + fetch-stats counters (`6f7eb215..1d77cd2d`). Built by recompiling three objects (`verify.cpp`, `expert_source.cpp`, `generate.cpp`) onto the int8 build objects — manifest in the deploy folder |
| serve / control | `main` (this tree), tip `f50eb8f6`: v0.1.45 port line + tokenizer optimizers `e9ff3756` (translate table) and `f50eb8f6` (rank-key dedup, **−28.7 MB RSS at load**, measured fresh-process) |
| config | `prod-config/PT3M-262k.sample.json` here; live file `prod/batch-test/PT3M-262k.json` (identical except the redacted MCP bearer token) |
| args (live) | `--max-context 262144 --kv int8 --batch 3 --spec 4 --pool-workers 12 --pcie-frac 0.20 --expert-cache auto --expert-profile …-learned.bin --expert-profile-save … --conversation-cache-mib 16384 --conversation-cache-slots 8 --kv-resident 20480 --stream-experts --pack iq3_s` |
| armed gates | `STRATA_NATIVE_GRAPH=1`, `STRATA_BATCH_MTP=1`, `STRATA_TAIL_SKIP_IN_BATCH=1`, `STRATA_PREFILL_CPU_SHARE_BATCH=1`, `STRATA_SLOT_COPY_DIRECT=1`, `STRATA_PREFILL_FIRST=0`, `STRATA_PREFILL_CPU_SHARE=auto`, int8 DPAS prompt experts on by default (`STRATA_PREFILL_MMQ=0` disables); fetch counters and all trace knobs off |

## Deploy chain of 2026-10-20

1. `deployments/diagnostics-20261020-013451` — binary swap to `1d77cd2d` (instr + fetch-stats build).
2. `deployments/deploy-020workers-20261020-110225` — baseline 20 pool workers / frac 0.20.
3. `deployments/workers12-20261020-120911` — config-only back to 12 workers (current baseline).
4. 18:16 — serve-side only: tokenizer commits deployed (engine binary and config untouched).

Each deploy has a guarded `rollback.sh`. Engine restarts happen only at idle.

## Measured today (single-stream suite, `--runs 1`, 12 workers / frac 0.20 / batch 3)

chat 67.7 tok/s (60.7–80.6) · agent session 62.4 (53.7–74.1), TTFT 2.99 s · 4K prompt 1,423 tok/s ·
16K prompt 1,803 tok/s · whole suite 42.8 s, 0 errors, 0 skips.

## Code-review findings open against this line (shared with the agent team in DISCUSSION.md)

- **Pool-guard tripwire (latent):** the exclusive expert-pool guard throws `std::runtime_error` on a collision;
  nothing between the throw site and `main` catches non-SYCL exceptions, so a collision would end the engine
  process instead of failing one request. 0 collisions observed in the live log since arming. Not a miscompute —
  an availability nit; a fix needs pool quiescence, not a blind catch.
- **Parity test not in CI:** the GPU parity test for the live direct-slot-copy path (`tests/slot_copy/`) is not
  wired into any CMake target and is env-gated; manual parity evidence exists (bitwise gate + 170 candidate
  vectors hash-equal), automated coverage does not.
- **Reproducibility coupling:** `STRATA_TAIL_SKIP_IN_BATCH=1` makes an slot's answers depend on which other slots
  are busy, and long greedy generations are not bit-stable run-to-run; A/B parity runs must fix the side load,
  not only the config.
- One earlier review alarm (missing slot invalidation on the yield path) was **withdrawn** — the invalidation is
  there; recorded so nobody re-investigates it.
