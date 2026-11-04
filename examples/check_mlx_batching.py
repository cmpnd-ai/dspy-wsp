"""Check whether MLXBackend's batched decoding computes the same model as unbatched reads.

Each prompt gets a greedy answer three ways:

  alone    one prompt at a time, so nothing is batched
  batched  all prompts decoding together, each prompt read alone first
  padded   all prompts together, with prompts read in padded groups of 8 (the backend's default)

Then a single unbatched forward pass reads each prompt plus its answer and records,
at every answer position, the most likely next token and its margin over the
runner-up (in nats). A greedy answer should always pick the most likely token.
Where it does not, a small margin means rounding noise; a large margin means the
batched computation differs from the model.

    python examples/check_mlx_batching.py            # 16 prompts, 512 tokens each
    python examples/check_mlx_batching.py --n 32 --tokens 1024

Prompts come from a saved MATH-500 run (`--prompts`), so nothing is downloaded but the model.
"""

import argparse
import json
import statistics
import threading
import time
from pathlib import Path

from dspy_wsp import MLXBackend

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
DEFAULT_PROMPTS = Path(__file__).parent / "runs/math500-qwen-mlx/final_check/original.jsonl"


def generate(backend: MLXBackend, ids: list[int], max_tokens: int) -> list[int]:
    """Greedy tokens for one prompt, submitted straight to the backend's batch."""
    from concurrent.futures import Future

    job = {"ids": ids, "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": None, "future": Future()}
    backend._submit(("gen", job))
    tokens, _ = job["future"].result()
    return tokens


def run_mode(backend: MLXBackend, prompts: list[list[int]], max_tokens: int, mode: str) -> tuple[list[list[int]], float]:
    prefill = 8 if mode == "padded" else 1
    backend._call(lambda: setattr(backend._gen, "prefill_batch_size", prefill))
    start = time.time()
    if mode == "alone":
        out = [generate(backend, p, max_tokens) for p in prompts]
    else:
        out: list = [None] * len(prompts)

        def work(i):
            out[i] = generate(backend, prompts[i], max_tokens)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(len(prompts))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    return out, time.time() - start


def reference(backend: MLXBackend, ids: list[int]):
    """Top-2 next-token predictions at every position, from one unbatched forward pass."""
    return backend._call(lambda: backend._logprobs(backend._read(ids, hidden=[], mlp=[], attn=[])[0], ids, 2)[1:])


def compare(backend: MLXBackend, prompts, answers) -> dict:
    margins_all, flips, first = [], [], []
    for p, a in zip(prompts, answers):
        ids = p + a
        top_ids, top_lp = reference(backend, ids)
        first_flip = None
        for j, tok in enumerate(a):
            pos = len(p) + j - 1  # the position whose prediction is answer token j
            margin = float(top_lp[pos, 0] - top_lp[pos, 1])
            margins_all.append(margin)
            if tok != int(top_ids[pos, 0]):
                flips.append(margin)
                if first_flip is None:
                    first_flip = j
        first.append(first_flip)
    return {
        "tokens": len(margins_all),
        "flips": len(flips),
        "flip_margins": sorted(round(m, 3) for m in flips),
        "max_flip_margin": round(max(flips), 3) if flips else None,
        "first_flip_token": first,
        "share_of_positions_below_0.1": round(sum(m < 0.1 for m in margins_all) / max(1, len(margins_all)), 4),
    }


def agreement(a: list[list[int]], b: list[list[int]]) -> list[int]:
    """How many leading tokens each pair of answers shares."""
    out = []
    for x, y in zip(a, b):
        k = 0
        while k < min(len(x), len(y)) and x[k] == y[k]:
            k += 1
        out.append(k)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16, help="prompts to test")
    parser.add_argument("--tokens", type=int, default=512, help="greedy tokens per answer")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS, help="a saved run's rows, with calls[0].prompt")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "runs/check_mlx_batching.json")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.prompts.open()][: args.n]
    texts = [r["calls"][0]["prompt"] for r in rows]
    backend = MLXBackend(MODEL, revision=REVISION, batch_size=args.n, max_tokens=args.tokens, temperature=0.0, chat_template_kwargs={"enable_thinking": True})
    prompts = [backend._encode(t) for t in texts]
    print(f"{len(prompts)} prompts, {min(map(len, prompts))} to {max(map(len, prompts))} tokens; {args.tokens} greedy tokens each", flush=True)

    answers, report = {}, {"n": len(prompts), "tokens": args.tokens}
    for mode in ("alone", "batched", "padded"):
        answers[mode], secs = run_mode(backend, prompts, args.tokens, mode)
        report[mode] = compare(backend, prompts, answers[mode])
        report[mode]["seconds"] = round(secs, 1)
        r = report[mode]
        print(f"{mode:8s} {r['seconds']:6.1f}s  flips {r['flips']:3d} of {r['tokens']}  largest flip margin {r['max_flip_margin']}", flush=True)
    for mode in ("batched", "padded"):
        report[f"{mode}_vs_alone_shared_tokens"] = agreement(answers[mode], answers["alone"])
        shared = report[f"{mode}_vs_alone_shared_tokens"]
        same = sum(x == y for x, y in zip(answers[mode], answers["alone"]))
        print(f"{mode} vs alone: median shared prefix {statistics.median(shared)} tokens, identical answers {same}/{len(shared)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
