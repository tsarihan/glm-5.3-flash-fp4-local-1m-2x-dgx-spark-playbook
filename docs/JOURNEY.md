# JOURNEY — from "it loads" to "it survives"

The timeline of getting `nvidia/GLM-5.3-Flash-FP4-Local` from a crashed container to a
permanent 1M-context lane on two DGX Sparks. All dates 2026.

## Oct 24 — the first day: five fixes, then a wall

Tedngai's measured 2×Spark recipe (target ~500K context, GMU 0.87) was the starting point,
retargeted to the model's native 1,048,576. Four problems fell quickly:

1. **The b12x loader needs io_uring.** Docker's default seccomp profile blocks it:
   `io_uring bounce initialization failed: Operation not permitted`.
   `--security-opt seccomp=unconfined` is the narrow fix (cheaper than `--privileged`).
2. **FlashInfer drift.** The fork calls `intrinsics.swizzle_block_scale`, removed on
   2026-10-17 (`c46a0b50`) when FlashInfer split the b12x intrinsics into a package. Building
   against current FlashInfer fails at load with `AttributeError`. Pin `--flashinfer-ref`
   before that commit. (Two images were built, `fi-0213747d` and `fi-b1f081e6`, and behave
   identically; the latter is what runs.)
3. **The default read mode hangs rank 1.** `--model-loader-extra-config` defaults to `auto`,
   which selects GDS plus a `SharedReadGroup` across the TP group. Rank 1 on the second node
   sat in that group for 3 minutes and never loaded. `read_mode: bounce` — every node reads
   its own shards — loads both in ~145 s.
4. **Upstream's launcher uses `--rm` containers.** Discarded immediately: a crashing
   `--rm` container deletes its own evidence.

Then the wall. Both ranks loaded weights cleanly — and about two seconds after "Loading
weights took ~145 s", MemAvailable on **both** nodes collapsed from ~20 GiB to **0–2 GiB**.

## Oct 24 (cont.) — five wrong suspects

The collapse was chased for most of a day:

| suspect | why it looked plausible | why it isn't |
|---|---|---|
| GMU too high | 121 GiB nodes, tight by design | identical dip at 0.85 and 0.82 |
| 1M context | the pool is the biggest allocation | **500K dips to 0 GiB the same way** |
| FlashInfer age | two pins were in play | both pins behave identically |
| Starting headroom | browsers/services eat memory | from 115 GiB free on both nodes, still 0 |
| KV pool / profiling batch | the usual suspects | dip is post-load, per-node, and eats *whatever* is free |

That last property was the tell. A fixed-size spike would leave a fixed remainder; this ate
all of it on both nodes independently — spark-2 hit 0 GiB while spark-1 was still loading.
The memtrace showed the growth was all "unaccounted" GPU allocations (91 → 114 GiB) while
anonymous memory stayed ~4 GiB, and the vLLM debug log contained the allocator confession:
`Released 12.18GiB of cached device memory under memory pressure`. The PyTorch caching
allocator was holding every block it had freed during b12x state preparation.

**Fix: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — one environment variable,
lowest free memory after load ~10 GiB on both nodes. The same report/fix later surfaced in
the model's HF discussions (#7/#8) and upstream (#388), which is why this README leads with
it.

A 2 GiB floor (`memguard.sh`) was kept anyway: two days of running showed the real steady-state
floor at 4–6 GiB on both nodes with multimodal resident.

## Oct 25 — multimodal at 1M: one more GiB of GMU

Enabling vision (`IMAGES=4 VIDEOS=1`) at GMU 0.82 computed 6.93 GiB of KV — less than the
7.48 GiB a single 1M request needs. The engine refused to start with a KV-shortfall error
that never says the word "vision" (identical failure shape to the old lane's vision trap;
the visual tower is unquantized in both weight sets). GMU 0.84 buys it back: KV 1,307,683
tokens, 1.25× oversubscription at 1M.

That day also produced the lane's accuracy check: a 994,019-token needle-in-haystack
(passphrases planted at 3% and 96% depth), **both found**, prefill 573 s (1,734 tok/s),
lowest node free memory 5 GiB — inside the guard floor with margin.

## Oct 29 — two upstream levers and one guard

A sweep of upstream found the fork unchanged since our build (still `8244d1c`), the checkpoint
unchanged (still `76348dd`), and three issue-tracker items that mattered:

- **#416 — recurrent checkpoint policy.** The default `auto` drops hybrid-Mamba recurrent
  state at request end, so repeated long prefixes re-prefill fully. `aligned` checkpoints it
  with the KV blocks. Measured: 165,859-token prompt, repeat TTFT 77.6 s → **3.2 s**; then at
  full scale, the 990K needle doc, 587.5 s → **7.7 s**, with both needles still answered
  correctly from the cached path. For iterative full-repo coding/review — same context, new
  question — this is the difference between a 10-minute and a 10-second turnaround.
- **#432 — adaptive speculative tokens.** First boot attempt crashed ~40 s in: the window
  setting alone turns on dynamic speculative decoding, and this commit then asserts a
  per-batch-size schedule exists. With the schedule supplied (`[[1,8,5]]`), measured gain is
  **+~4% decode** (33 vs 31.6–31.9 tok/s), not the +22–42% in circulating community figures —
  published here because honest calibration beats repeating the headline.
- **#409 — the freeze.** Under sustained agent traffic the engine can wedge (~once/12–24 h):
  the watchdog kills EngineCore, workers linger at ~85 GB RSS, containers stay `Up`, the API
  is dead. memguard can't see it (memory is fine), so `apiwatch.sh` watches the API itself
  and does the stop/drain/relaunch cycle per node. Armed after first 200, three consecutive
  failures to act, 60 s polls.

## Oct 29 — validated as an agent backend

The last open question was whether the lane survives being driven the way it's actually meant
to be used: a CLI agent harness talking Anthropic protocol with real filesystem tools against
a big repo. One solo council member (model-council MCP, the `claude-cli` harness pointed at
this server's native `/v1/messages`) crawled a 34-crate Rust workspace read-only and produced
an architecture map whose file:line citations spot-checked exact — ~8.5 min, 2.14M prompt
tokens, 79% served from prefix cache across the loop.

The lane is now the default council backend for local review work.
