# Adaptive MTP vs fixed MTP-1 — measured, this build (`8244d1c`)

All single-stream, server-side `/metrics` spec-decode counters for token truth (per-chunk
client counting understates multi-token MTP steps; this fork's reasoning deltas use the key
`reasoning`, not `reasoning_content`).

| config | acceptance | avg draft depth | decode, long answers |
|---|---|---|---|
| fixed MTP, `num_speculative_tokens 1` | 77.9% | — | 31.6–31.9 tok/s |
| **adaptive**: `num 5, num_speculative_tokens_per_batch_size [[1,8,5]], window 8, initial 5` | 52–54% | ~2.2 (controller-converged) | **~33 tok/s (+~4%)** |

Prefill unchanged within noise: 990K prompt, 1,685 tok/s adaptive vs 1,734 tok/s on the
2026-10-25 fixed-MTP run.

**Calibration against the community literature:** figures of +22% (hard prose) to +32–42%
(mixed single stream) circulate for adaptive spec tokens on this model/hardware class. We do
not reproduce that; we reproduce +4%. Possible sources of divergence: their baseline k, batch
shape, or fork vintage. Publishing our number because a same-harness A/B beats quoting theirs.

**Config trap (this commit):** `adaptive_speculative_tokens_window` alone enables dynamic
spec decode, and `_set_cudagraph_sizes` then asserts a schedule exists — window-without-
schedule dies ~40 s into boot (pydantic `ValidationError ... Assertion failed`) on both
ranks. Supply all three fields. `ADAPTIVE=0` in `serve.sh` rolls back to fixed MTP.
