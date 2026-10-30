# Agent-backend validation (2026-10-29)

The server answers the Anthropic `/v1/messages` API natively, so a `claude` CLI harness can
be pointed straight at it — no proxy, no protocol shim. Via
[model-council-mcp](https://github.com/tsarihan/model-council-mcp), one solo
`claude-cli`-harness member backed by this GLM lane (`claude-cli-vllm-dgx-spark-1`) ran a
whole-repo read-only review of a 34-crate Rust workspace, with full-repo tool access
(Read/Grep/Glob confined to the repo root) and file output.

| | |
|---|---|
| wall time | ~8.5 min |
| prompt tokens processed across the agent loop | 2,144,403 |
| served from prefix cache | **79%** (1,697,536 hits) |
| output | architecture map + fact answers, ~15–25 KB, saved under `council-reviews/` |
| citation audit | file:line citations spot-checked against the repo — **exact** (e.g. `score_against_ground_truth` at the cited line of `benchmark/src/lib.rs`) |
| read-only contract | held — no repo file written; only its own scratch outputs |

The 79% prefix-cache hit rate across an agentic crawl is the aligned recurrent-checkpoint
config (README finding #2) doing its job on real agent traffic: each loop iteration
re-prefills only the delta, and the Mamba state survives between turns.
