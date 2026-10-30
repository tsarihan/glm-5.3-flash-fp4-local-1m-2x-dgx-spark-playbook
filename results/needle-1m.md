# 1M-class needle-in-haystack, cold vs cached

Two runs of the same recipe: a deterministic numbered log with two passphrases planted
(`scripts/needle.py`, seed-pinned). Question at the tail asks for both, in order, nothing
else.

## 2026-10-25 — fixed MTP-1, GMU 0.84, multimodal resident

- prompt tokens: **994,019** (needles at 3% and 96% depth)
- pass 1 (cold): prefill 573 s = **1,734 tok/s**; needle A 3% **PASS**; needle B 96% **PASS**
- lowest node free memory during: rank-0 node 5 GiB, rank-1 node 8 GiB

## 2026-10-29 — adaptive MTP + `--recurrent-checkpoint-policy aligned`, same lane

- prompt tokens: **990,010** (deterministic generator, 18,650 records; exact count via
  `POST /tokenize`)
- pass 1 (cold): **TTFT 587.5 s** (1,685 tok/s) — needle A **PASS**, needle B **PASS**
- pass 2 (byte-identical repeat): **TTFT 7.7 s (76×)** — needle A **PASS**, needle B **PASS**
  on the cached path

The cached pass answers *correctly*, not just quickly: both passphrases are returned from a
response whose prompt was served from the prefix cache. Cache hits are not served stale here.

## Mid-scale calibration point (same day, 166K)

- 165,859-token prompt: cold TTFT 77.6 s (2,140 tok/s); identical repeat **3.2 s**.

## Answer text (pass 2, cached), tail

    Looking through the log:
    1. After Record 560: "IMPORTANT NOTE: the first vault passphrase is VERMILION-OTTER-4471."
    2. After Record 17904: "IMPORTANT NOTE: the second vault passphrase is COBALT-HERON-9038."
    ...VERMILION-OTTER-4471
    COBALT-HERON-9038

Passphrase values differ per generated document (they are constants in this run's generator);
what matters is that both depths resolve on both passes.
