"""GLM validation probe: decode tok/s probe + prefix-cache A/B against rank-0 node :8000.

usage:
  python3 probe.py decode                 # long-answer decode speed probe (short prompt)
  python3 probe.py bigtxt TARGET_TOKENS   # write deterministic big prompt to bigprompt.txt
  python3 probe.py prefix                 # send bigprompt.txt twice, compare TTFT (cache A/B)
"""
import json, sys, time, urllib.request

BASE = __import__("os").environ.get("GLM_API", "http://127.0.0.1:8000") + "/v1/chat/completions"  # GLM_API = http://<rank-0 mgmt IP>:8000
MODEL = "glm-5.3-flash-fp4-local"
PARA = ("The quick brown fox jumps over the lazy dog while the committee reviews the quarterly "
        "appendix on distributed inference scheduling. Every sentence in this document is filler "
        "text used only to occupy context space consistently across repeated runs. ")


def stream(prompt, max_tokens):
    body = {"model": MODEL, "max_tokens": max_tokens, "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}], "stream": True}
    req = urllib.request.Request(BASE, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer local"})
    t0 = time.time(); ttft = None; ntok = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data:") or b"[DONE]" in line:
                continue
            d = json.loads(line[5:])
            ch = d.get("choices") or [{}]
            delta = ch[0].get("delta") or {}
            if (delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")):
                if ttft is None:
                    ttft = time.time() - t0
                ntok += 1
    dt = time.time() - t0
    return ttft or dt, ntok, dt


if __name__ == "__main__" and sys.argv[1] == "decode":
    prompt = ("Write a detailed engineering essay (at least 1500 words) about how speculative "
              "decoding works in mixture-of-experts language models, covering draft/verify loops, "
              "acceptance rates, and batching effects.")
    ttft, ntok, dt = stream(prompt, 2048)
    print(f"decode: {ntok} tokens in {dt:.1f}s = {ntok/max(dt - ttft, 0.01):.1f} tok/s (ttft {ttft:.1f}s)")

elif __name__ == "__main__" and sys.argv[1] == "bigtxt":
    target = int(sys.argv[2])
    est = target * 4 + 4000          # ~4 chars/token for prose, slack
    text = (PARA * (est // len(PARA) + 1))[:est]
    text += ("\n\nThe verification code mentioned once in this document is ZEBRA-FACET-2210.\n"
             "Question: what is the verification code mentioned in this document? Answer with the code only.")
    open("bigprompt.txt", "w").write(text)
    print(f"wrote bigprompt.txt: {len(text)} chars (~{len(text)//4} tokens rough)")

elif __name__ == "__main__" and sys.argv[1] == "prefix":
    text = open("bigprompt.txt").read()
    for i in (1, 2):
        ttft, ntok, dt = stream(text, 40)
        print(f"pass {i}: ttft={ttft:.1f}s total={dt:.1f}s")
