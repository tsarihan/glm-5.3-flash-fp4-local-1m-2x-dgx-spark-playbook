# Stream ladder — concurrent decode on the 1M multimodal lane (2026-10-30)

First concurrency measurement on this config. `scripts/probe.py`-style streaming, one
OpenAI `chat/completions` request per stream, **unique prompt per stream per rung** (rung +
worker + wall-clock nonce) so prefix caching cannot bleed between rungs, `temperature 0`,
`max_tokens 256`, streaming with `include_usage`. Prompts are short (~120 tokens), so this
is a decode-dominated ladder, not a shared-prefill ladder — see the caveat at the bottom.

| streams | per-stream tok/s (avg) | per-stream spread | aggregate tok/s | scaling | TTFT avg |
|---|---|---|---|---|---|
| 1 | 33.9 | — | 32.2 | 1.00× | 0.39 s |
| 2 | 26.6 | 23.9–29.2 | 46.6 | 1.45× | 0.74 s |
| 4 | 18.2 | 17.8–18.6 | 67.5 | 2.10× | 0.65 s |
| 8 | 13.2 | 12.9–13.7 | 99.1 | 3.08× | 0.82 s |

All requests completed at every rung: no preemption, no engine error, TTFT under 1 s
throughout.

## Adaptive MTP under batch load

Spec-decode counters (`/metrics`, deltas over an isolated n=8 run, so no contamination from
earlier traffic):

| | single-stream (earlier adaptive runs) | n=8 (this ladder) |
|---|---|---|
| MTP acceptance | 52–54% | **59.8%** |
| avg draft depth | ~2.2 | 2.00 |

Concurrency does **not** collapse adaptive MTP: at batch 8 the controller settled at depth
~2 (the `[[1,8,5]]` schedule permits 5 for batch sizes 1–8, so it chooses less than its
ceiling) and acceptance *improved* on this content. The per-batch-size schedule doing its
job is the reason finding #4's config shape matters — without the schedule, boot dies
instead.

## Reading the numbers

- Aggregate scaling is sublinear (1 → 1.45× → 2.10× → 3.08×): GB10 decode is
  bandwidth-bound, so splitting the bus across 8 streams roughly triples each stream's
  per-token time. Going 1 → 8 streams costs ~2.6× per stream.
- This is the reconciliation with circulating community figures of 60–116 tok/s for this
  model: those are **multi-stream aggregates** at exactly this shape. 99 tok/s aggregate at
  8 streams sits inside that band; honest single-stream is ~33.
- For agent workloads: one stream is fastest per-task (README's worker-count guidance
  stands), but if you are feeding a small fleet of agents, 4 streams at ~18 tok/s each is a
  reasonable balance — half the per-stream speed of solo, double the aggregate.

## Caveat

Short prompts only. A ladder where every stream also carries a long prefill is a different
test: prefills serialize against the 8,192-token chunk budget, and the recurrent-checkpoint
cache (`finding #2`) would interact with 8 concurrent long-context requests in ways this
ladder does not touch. That variant is untested.
