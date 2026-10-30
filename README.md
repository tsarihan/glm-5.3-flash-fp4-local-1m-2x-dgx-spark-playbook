# GLM-5.3-Flash **FP4-Local** at **1M context** on 2× DGX Spark (GB10)

Serving NVIDIA's `nvidia/GLM-5.3-Flash-FP4-Local` (ModelOpt mixed-precision: NVFP4 experts +
MXFP8 attention) TP=2 across two DGX Spark (GB10, `sm_121`) nodes **at the model's native
1,048,576-token context**, on the `b12x` vLLM fork — with adaptive MTP speculative decoding,
prefix caching that actually survives across requests on a hybrid-Mamba model, multimodal
(images + video), and the operational guards this lane needs to stay up for weeks at a time.

**The journey — the post-load memory dip that took both nodes to 0 GiB free, five ruled-out
hypotheses, and the one environment variable that fixed it:** [docs/JOURNEY.md](docs/JOURNEY.md)

**The predecessor playbook** (RedHatAI NVFP4 weights, stock vLLM, 244K context, three lanes
attempted / one works — SGLang and GGUF failed for architectural reasons that still stand):
https://github.com/tsarihan/glm-5.3-flash-nvfp4-2x-dgx-spark-playbook

**Companion repos (same hardware, different models):**
- https://github.com/tsarihan/qwen3.8-flash-next-nvfp4-2x-dgx-spark-playbook
- https://github.com/tsarihan/deepseek-v4-flash-0731-nvfp4-2x-dgx-spark-playbook
- https://github.com/tsarihan/qwen3.8-27b-dgx-spark-gb10-playbook

© 2026 Tom Sarihan, Desnet AI LLC. Apache-2.0 (see `LICENSE`, `NOTICE`).

*Author accounts: GitHub [@tsarihan](https://github.com/tsarihan), Hugging Face [@tomsarihan](https://huggingface.co/tomsarihan). The Hugging Face account `tsarihan` is a different person and is unrelated to this work.*

---

## The short version

This lane **supersedes the RedHatAI/stock-vLLM playbook** on the same two nodes:

| | old lane (RedHatAI NVFP4, stock vLLM) | **this lane (NVIDIA FP4-Local, b12x fork)** |
|---|---|---|
| context | 244,224 | **1,048,576 (4.3×)** |
| decode, single stream | 21.45 tok/s (MTP-4) | **31.6–33 tok/s** |
| CUDA graphs | none (`--enforce-eager` was mandatory) | **on** (B12X attention backend) |
| repeat 990K-token prefix | full re-prefill, ~575 s | **7.7 s (76×)** |
| multimodal | 4 images, 0 video | **4 images + 1 video** |

Measured headline numbers (single stream, these nodes, this build):

| metric | value |
|---|---|
| decode, adaptive MTP | **~33 tok/s** (52–54% acceptance, controller settles ~2.2 avg depth) |
| decode, fixed MTP-1 (rollback config) | 31.6–31.9 tok/s (77.9% acceptance) |
| prefill throughput | 1,685–2,140 tok/s (990K and 166K prompts) |
| 1M-class needle-in-haystack | **2/2 passphrases found** @ 990,010 prompt tokens, needles at 3% and 96% depth — **on both the cold pass and the cached pass** |
| cold 990K prefill | 587.5 s TTFT |
| **identical repeat** | **7.7 s TTFT** (prefix cache + `--recurrent-checkpoint-policy aligned`) |
| vision | image and video requests pass, direct and through an OpenAI/Anthropic-protocol gateway |

**The most useful things in this repo are the traps**: one env var without which both nodes
dive to 0 GiB free seconds after loading
([#1](#1-pytorch_cuda_alloc_confexpandable_segmentstrue-is-load-bearing)); an adaptive-MTP
config shape that dies at startup with a pydantic assert unless you supply a per-batch-size
schedule ([#4](#4-adaptive-mtp-needs-its-per-batch-size-schedule-at-this-commit)); and a fork
whose reasoning stream uses a nonstandard delta key, which makes naive clients report 0 tokens
([#7](#7-this-forks-reasoning-stream-uses-a-nonstandard-delta-key)).

---

## Reproducibility — everything is pinned

| component | pin |
|---|---|
| Model | `nvidia/GLM-5.3-Flash-FP4-Local`, revision `76348dd`, sha256-verified against the HF LFS hashes |
| Quantization | NVIDIA ModelOpt `MIXED_PRECISION` — see [the quant layout](#the-quant-layout-nvfp4-not-mxfp4) |
| vLLM | `local-inference-lab/vllm` fork, branch `dev/karmic-kraken` @ **`8244d1c`** (b12x) |
| Build harness | `eugr/spark-vllm-docker` @ `0bdc9c9`, `--exp-b12x --rebuild-vllm --rebuild-flashinfer` |
| FlashInfer | pin **`--flashinfer-ref`** to a commit *before* `c46a0b50` (2026-10-17) — image tag `vllm-node-b12x:fi-b1f081e6` |
| GPU arch | `sm_121` (GB10), `CUTE_DSL_ARCH=sm_121a` |
| Attention / MoE / linear backends | `B12X` / `b12x` / `b12x` |
| KV cache | `fp8`; mamba cache `align`; prefix caching on; `--recurrent-checkpoint-policy aligned` |
| Speculative decoding | MTP, **adaptive**: `num 5, schedule [[1,8,5]], window 8, initial 5` |
| Allocator | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — **required** |
| Interconnect | RoCE over the node's second fabric (`enP2p1s0f0np0`), NCCL over IB/RoCE v2 |
| Context / batching | 1,048,576 · 8 seqs · chunk 8,192 · block 256 |
| GPU memory utilization | 0.85 text-only; **0.84 with vision enabled** |
| Flags lineage | follows tedngai's `glm-5.3-flash-tuned.yaml` recipe (`a810bf2`), divergences documented in `scripts/serve.sh` |

---

## The quant layout: NVFP4, not MXFP4

`quant_method: modelopt`, `quant_algo: MIXED_PRECISION`. The served checkpoint is four formats
targeted per-module, not one blanket format:

| modules | algo | where |
|---|---|---|
| 42 | **NVFP4** (W4A4, group-16, `csf_32x2_v1` scales) | MoE experts, lower layers |
| 77 | **W4A16_NVFP4** (NVFP4 weights, bf16 activations) | MoE experts, upper layers |
| 170 | FP8_PB_WO (FP8 per-block 128×128, weight-only) | shared experts |
| 205 | **MXFP8** (microscaling FP8, 1×32 blocks) | attention q/k/o, lm_head, dense MLPs |

There is **zero MXFP4** in this checkpoint. Community builds of this model family floating
around are mostly MXFP4 or single-format conversions with no published calibration set or
accuracy data — this NVIDIA build is the one with disclosed methodology, and it is why we
serve it. vLLM side: `--quantization modelopt_mixed`. KV cache is left at `fp8`
(`kv_cache_scheme` is null in the checkpoint).

---

## Findings

### 1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is load-bearing

Without it, about two seconds after "Loading weights took ~145 s", MemAvailable falls on
**both** nodes from ~20 GiB to **0–2 GiB** — at any GMU from 0.82–0.85, at any context length,
text-only or multimodal. The memtrace shows the growth is all GPU allocations held by the
PyTorch caching allocator during b12x state/bind preparation ("Released 12.18GiB of cached
device memory under memory pressure" appears in the debug log). With the env var, the lowest
free memory after load is ~10 GiB on both nodes. This exact failure and fix are also in HF
`nvidia/GLM-5.3-Flash-FP4-Local` discussion #7/#8 and in `eugr/spark-vllm-docker` #388.

Five hypotheses were ruled out before this one: GMU, context length, FlashInfer age, starting
headroom, and the KV pool size. It is a post-load allocator behaviour, per-node, that eats
*whatever* is free.

### 2. `--recurrent-checkpoint-policy aligned` is the 800K-workflow lever

GLM-5.3 is hybrid (Mamba + attention). With the default `auto` policy, the recurrent state is
dropped at request end, so a repeated long prefix re-prefills from scratch. `aligned`
checkpoints the recurrent state alongside KV blocks, so identical prefixes come back:

| prompt | cold TTFT | identical repeat TTFT |
|---|---|---|
| 165,859 tok | 77.6 s (2,140 tok/s) | **3.2 s** |
| 990,010 tok (needle doc) | 587.5 s (1,685 tok/s) | **7.7 s** |

Both needle passphrases (3% and 96% depth) were answered correctly on the cached path too —
cache hits are not served stale. The companion flags in `serve.sh`:
`--mamba-cache-mode align --enable-prefix-caching --recurrent-checkpoint-policy aligned`.
Caveat from upstream (eugr #416): the cache busts *wholesale* if anything mid-prefix changes.

### 3. Adaptive MTP helps less than advertised here — and that's the honest number

A widely-quoted community figure claims +22–42% single-stream from adaptive speculative
tokens on this model. Measured on this build, vs our own fixed MTP-1 baseline:

| config | acceptance | decode |
|---|---|---|
| fixed MTP-1 | 77.9% | 31.6–31.9 tok/s |
| **adaptive** (num 5, sched `[[1,8,5]]`, win 8, init 5) | 52–54% | **~33 tok/s (+~4%)** |

Prefill is unaffected (990K at 1,685 vs 1,734 tok/s on the earlier fixed-MTP run — within
noise). It is kept because it adapts draft depth to content, not because it is a headline
number. `ADAPTIVE=0 ./scripts/serve.sh` rolls back.

### 4. Adaptive MTP needs its per-batch-size schedule at this commit

At `8244d1c`, `uses_dynamic_speculative_decoding()` is true from the **window** setting alone,
and `_set_cudagraph_sizes` then runs `assert schedule is not None`. A config with
`adaptive_speculative_tokens_window` but **no** `num_speculative_tokens_per_batch_size` dies
~40 s into boot with a pydantic `ValidationError ... Assertion failed` on both ranks. Supply
all three — window, initial, and schedule — as `serve.sh` does.

### 5. 1M context at multimodal needs GMU 0.84, not 0.82

The vision encoder budget costs ~2.5 GiB of KV. At `GMU=0.82` the KV pool computes to
6.93 GiB < the 7.48 GiB one full 1M request needs (max 958,464) — the engine refuses to start
with a "KV shortfall" error that never mentions vision (same failure shape as the old lane).
At 0.84: KV **1,307,683 tokens (1.25× at 1M)**. Text-only at 0.82 gives 1,327,926 (1.27×).
Post-load free memory with 4 images + 1 video enabled: rank-0 node ~6–7 GiB, rank-1 node ~9 GiB —
that's why `memguard.sh` exists.

### 6. The engine can freeze dead while every container looks healthy

Under sustained coding-agent traffic (~once per 12–24 h, any context length — upstream
eugr #409), an execute-model step can freeze; the watchdog kills EngineCore, but the worker
processes linger at ~85 GB RSS. Containers: `Up`. Memory: fine (memguard can't see it). API:
dead. `apiwatch.sh` polls the API from each node and, after three consecutive failures,
stops its own rank's container (`docker stop`, never `rm`), waits for memory to drain,
relaunches, and re-arms. It polls `/v1/models` — a metadata endpoint that keeps answering
during a 10-minute prefill, so long prefills cannot false-trigger it. **Inverse caveat worth
knowing:** if you operate this, confirm your failure mode actually kills the metadata
endpoint; on some vLLM builds the metadata route keeps answering with a dead engine (the old
playbook's health-check trap, in reverse).

### 7. This fork's reasoning stream uses a nonstandard delta key

SSE deltas carry `delta.reasoning`, **not** `delta.reasoning_content`. A client that only
counts the latter reports **0 tokens in 63 s** on a healthy server. Also: multi-token MTP
steps emit several tokens per chunk, so chunk-counting understates tok/s — use the server's
`/metrics` spec-decode counters (`vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total`)
for truth. Both are handled in `scripts/probe.py`.

### 8. FlashInfer must predate `c46a0b50` (2026-10-17)

Newer FlashInfer split the b12x intrinsics into a package and dropped
`intrinsics.swizzle_block_scale`, which this vLLM fork calls at load (`AttributeError`).
Build with `--flashinfer-ref 0213747db` (or any pre-c46a0b50 commit). The served image here
keeps `b1f081e6`.

### 9. The b12x loader wants two non-obvious settings

- `--security-opt seccomp=unconfined` — the loader uses io_uring, which docker's default
  seccomp profile blocks ("io_uring bounce initialization failed"). Narrower than
  `--privileged`.
- `--model-loader-extra-config '{"read_mode":"bounce"}'` — the default `auto` picks GDS + a
  `SharedReadGroup` across the TP group, which **hung rank 1 for 3 minutes** (it never
  loaded). `bounce` = each node reads its own local shards itself.

And one from the operator's manual that bites every session: **never pass
`--limit-mm-per-prompt` JSON through ssh** — the quoting mangles it. Pass `IMAGES`/`VIDEOS`
numbers to the script instead.

### 10. Validated end-to-end as an agent backend, not just a completion endpoint

The server answers Anthropic `/v1/messages` natively, so a `claude` CLI agent harness can run
against it with real tools (Read/Grep/Glob) and no proxy layer. A whole-repo read-only
architecture review of a 34-crate Rust workspace (file:line citations spot-checked exact)
completed in ~8.5 min through that path, with the engine processing 2.14M prompt tokens at
**79% prefix-cache hits** across the agent loop — the aligned-checkpoint config doing exactly
what finding #2 says, under real agent traffic.

---

## Serving performance notes

Single-stream honest numbers are above; for calibration, published community figures of
60–116 tok/s for this model are **multi-stream aggregates**, and one much-cited 47.7 tok/s
figure is a **4× GB10 TP=4** run on a different (official NVFP4) image — different hardware
count, different build. On two GB10 nodes, single stream, at 1M context with vision resident,
~33 tok/s decode and ~1,700–2,100 tok/s prefill is the reproducible envelope. The measured
1/2/4/8-stream ladder (99 tok/s aggregate at 8 streams; adaptive MTP acceptance *rises* to
~60% at batch 8) is in [results/stream-ladder.md](results/stream-ladder.md).

`--gpu-memory-utilization 0.85` is the proven ceiling on these nodes; 0.90 wedged one. The
dip in finding #1 eats whatever is free, so free-memory floor monitoring (`memguard.sh`)
stays on permanently, not just during bring-up.

---

## Operational design

- **Named, kept containers.** No `--rm`, and `serve.sh` refuses to relaunch over an existing
  container. Dead containers are evidence; stop them, inspect them, delete on your own
  schedule.
- **Per-node guards.** `memguard.sh` (2 s poll, 2 GiB floor, logs the running minimum — the
  numbers in finding #5 came from its logs) and `apiwatch.sh` (finding #6) run on **both**
  nodes, each watching its own rank. apiwatch arms only after the first HTTP 200 so it never
  fights a normal multi-minute boot.
- **The API head is rank 0 only.** Anything on the second node must call the rank-0 address;
  `localhost` there connects to nothing. (This silently voided a 200-instance benchmark run
  on the old lane.)

## Setup

```bash
# both nodes: weights at /data/models/glm-5.3-flash-fp4-local-nvidia (sha256-verified),
# image vllm-node-b12x:fi-b1f081e6 built per the pin table above.
# NODE0_FABRIC / NODE1_FABRIC = the two nodes' RoCE fabric IPs;
# FABRIC_SUBNET = that fabric's CIDR; NODE0_MGMT = rank-0's management IP.

export NODE0_FABRIC=... NODE1_FABRIC=... FABRIC_SUBNET=.../24

# text-only, 1M, adaptive MTP (defaults):
NODE_RANK=0 ./scripts/serve.sh    # on rank 0
NODE_RANK=1 ./scripts/serve.sh    # on rank 1

# live multimodal lane (what we actually run):
NODE_RANK=0 GMU=0.84 IMAGES=4 VIDEOS=1 MAXLEN=1048576 ./scripts/serve.sh
NODE_RANK=1 GMU=0.84 IMAGES=4 VIDEOS=1 MAXLEN=1048576 ./scripts/serve.sh

# guards, each node, after launch:
./scripts/memguard.sh $(cat ~/llmruntimes/glm53-fp4local/last-container-r$NODE_RANK)
NODE_RANK=0 nohup ./scripts/apiwatch.sh >/dev/null 2>&1 &   # NODE_RANK=1 on the other node
```

Tunables (env): `MAXLEN GMU SEQS CHUNK BLOCK ADAPTIVE MTP MTP_WIN MTP_INIT MTP_SCHED
IMAGES VIDEOS READ_MODE PORT IMAGE MODEL EXTRA_ENV`.

## Reproducing the measurements

```bash
export GLM_API=http://<rank-0 mgmt IP>:8000

# decode probe + prefix-cache A/B on a synthetic big prompt
python3 scripts/probe.py decode
python3 scripts/probe.py bigtxt 160000 && python3 scripts/probe.py prefix

# 1M needle-in-haystack (deterministic; ~990K tokens, needles at 3% / 96% depth):
python3 scripts/needle.py gen 18650     # seed-pinned; writes needle-1m.txt next to itself
python3 scripts/needle.py tok           # exact prompt tokens via POST /tokenize
python3 scripts/needle.py ask 1         # cold pass (~10 min)
python3 scripts/needle.py ask 2         # identical repeat — cached-path TTFT
```

Raw logs and numbers: [results/](results/).

## Credits

GLM-5.3-Flash by [Z.ai](https://z.ai). FP4-Local quantization by **NVIDIA** (ModelOpt). The
b12x vLLM fork by local-inference-lab; build harness and recipe lineage by
[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) and tedngai's measured
2×Spark recipe. Serving: vLLM. Agent-loop validation: model-council MCP
([tsarihan/model-council-mcp](https://github.com/tsarihan/model-council-mcp)).
