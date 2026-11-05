"""Replication gate: the edits parser rebuilds the paper's hand-written network exactly, and our evaluator scores it
like the paper (53.11, SE 1.36, on 1,000 episodes)."""
import sys
import time

import numpy as np

import seaquest as S
from dspy_wsp.edit_gepa import language

if __name__ == "__main__":
    published = S.load_npz(S.DATA / "seaquest_hand.npz")
    net, skipped = language.apply(S.seed_program(), S.SHAPES)
    diff = max(float(np.abs(net[k] - published[k]).max()) for k in S.SHAPES)
    print(f"parser: {len(skipped)} skipped statements, max |rebuilt - published| = {diff}")
    assert not skipped and diff == 0.0, skipped
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    t = time.time()
    R = np.array([e["return"] for e in S.run_episodes(published, S.TEST[:n], keep_frames=0)], float)
    print(f"hand-written network on {n} test episodes: {R.mean():.2f} (SE {R.std(ddof=1) / np.sqrt(n):.2f}); "
          f"paper: 53.11 (SE 1.36); {time.time() - t:.1f}s")
