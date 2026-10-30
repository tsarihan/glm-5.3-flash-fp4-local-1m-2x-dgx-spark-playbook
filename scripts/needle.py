"""1M-token needle-in-haystack generator + sender for GLM FP4-Local on spark-1:8000.

Recreates the 2026-10-29 11:21 run shape (needle-glm-1m.txt):
  994,019 prompt tokens, filler of numbered Records, passphrases planted after
  Record 873 (~3% depth) and Record 27967 (~96% depth), question at the very end.

usage:
  python3 needle.py gen [records]          # write needle-1m.txt (deterministic)
  python3 needle.py tok                    # exact token count via POST /tokenize
  python3 needle.py ask [tag]              # send it streaming; print TTFT, usage, answer tail
"""
import json, random, re, sys, time, urllib.request

BASE = __import__("os").environ.get("GLM_API", "http://127.0.0.1:8000")  # GLM_API = http://<rank-0 mgmt IP>:8000
MODEL = "glm-5.3-flash-fp4-local"
FILE = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "needle-1m.txt")

# needle depths as fractions of the record count (match the 11:21 run's 3% / 96% depths)
NEEDLE_A_DEPTH = 0.030
NEEDLE_B_DEPTH = 0.960
NEEDLE_A = 'IMPORTANT NOTE: the first vault passphrase is VERMILION-OTTER-4471.'
NEEDLE_B = 'IMPORTANT NOTE: the second vault passphrase is COBALT-HERON-9038.'
QUESTION = ("\n\nQuestion for the reader: two IMPORTANT NOTE lines are hidden somewhere in the "
            "log above. Each one names a vault passphrase. Read the log, find both notes, and "
            "reply with both passphrases, first then second, and nothing else.")

SUBJ = ["Inventory audit", "Cache line", "Shipment batch", "Memory page", "Kernel thread",
        "Disk sector", "Network packet", "Ledger entry", "Checkpoint frame", "Sensor sample"]
VERB = ["verified against", "committed to", "flushed from", "migrated into", "compacted under",
        "rehashed beside", "archived with", "spooled through", "validated by", "rotated past"]
OBJ = ["the primary journal", "the standby ledger", "the cold-store volume", "the quorum log",
       "the staging buffer", "the replica set", "the manifest table", "the audit chain",
       "the rolling snapshot", "the parity stripe"]
TAIL = ["no anomalies observed.", "checksums matched on first pass.", "latency remained nominal.",
        "all counters stayed within tolerance.", "no operator action required.",
        "retention policy applied cleanly.", "sequence numbers continuous.", "parity bits intact."]
NEEDLE_RE = re.compile(r"VERMILION-OTTER-4471|COBALT-HERON-9038")


def gen(n_records):
    rng = random.Random(20261029)
    a_pos = round(NEEDLE_A_DEPTH * n_records)
    b_pos = round(NEEDLE_B_DEPTH * n_records)
    out = ["SYSTEM LOG BEGIN - routine batch reconciliation, deterministic replay.\n"]
    for i in range(n_records):
        parts = [f"Record {i}:"]
        for _ in range(rng.randint(2, 3)):
            parts.append(f"{rng.choice(SUBJ)} {i:05d}-{rng.randrange(97):02d} "
                         f"{rng.choice(VERB)} {rng.choice(OBJ)}; {rng.choice(TAIL)}")
        out.append(" ".join(parts) + "\n")
        if i == a_pos:
            out.append(NEEDLE_A + "\n")
        if i == b_pos:
            out.append(NEEDLE_B + "\n")
    out.append("SYSTEM LOG END.\n")
    out.append(QUESTION + "\n")
    text = "".join(out)
    with open(FILE, "w") as f:
        f.write(text)
    print(f"wrote {FILE}: {len(text):,} chars, {n_records} records, "
          f"needles after records {a_pos}/{b_pos}")


def tok():
    text = open(FILE).read()
    body = {"model": MODEL, "prompt": text}
    req = urllib.request.Request(BASE + "/tokenize", json.dumps(body).encode(),
                                 {"Content-Type": "application/json",
                                  "Authorization": "Bearer local"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    print(f"exact prompt tokens: {d['count']:,}")


def ask(tag):
    text = open(FILE).read()
    body = {"model": MODEL, "max_tokens": 256, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json",
                                  "Authorization": "Bearer local"})
    t0 = time.time(); ttft = None; answer = ""; usage = None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            line = line[5:].strip()
            if line == b"[DONE]":
                break
            d = json.loads(line)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or ""
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    answer += piece
    dt = time.time() - t0
    a = NEEDLE_RE.findall(answer)
    ok_a, ok_b = "VERMILION-OTTER-4471" in a, "COBALT-HERON-9038" in a
    print(f"pass {tag}: ttft={ttft:.1f}s total={dt:.1f}s "
          f"usage={json.dumps(usage) if usage else 'n/a'}")
    print(f"needleA(3%)={'PASS' if ok_a else 'FAIL'} needleB(96%)={'PASS' if ok_b else 'FAIL'}")
    print(f"answer tail: {answer[-400:]}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen":
        gen(int(sys.argv[2]) if len(sys.argv) > 2 else 29132)
    elif cmd == "tok":
        tok()
    elif cmd == "ask":
        ask(sys.argv[2] if len(sys.argv) > 2 else "run")
    else:
        sys.exit(f"unknown cmd {cmd}")
