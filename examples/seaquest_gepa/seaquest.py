"""MinAtar Seaquest for EditGEPA: the paper's Exp 5 network and input, seeded episodes, feedback text, and prompts.

Policy: 600 binary inputs (10x10 grid, 6 channels) -> 32 tanh -> 32 tanh -> 6 logits, actions sampled from the softmax,
sticky actions 0.1 (MinAtar's wrapper), episodes cut at 10,000 steps. An instance is a group of 8 episode seeds.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from minatar import Environment

from dspy_wsp.edit_gepa import language

HERE = Path(__file__).parent
DATA = HERE / "data"  # fetch_data.py
SHAPES = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}
ACTIONS = ["no-op", "left", "up", "right", "down", "fire"]
MAX_STEPS = 10_000
# the paper's 6 channels <- MinAtar seaquest channels (sub_front 0, sub_back 1, friendly_bullet 2, trail 3,
# enemy_bullet 4, enemy_fish 5, enemy_sub 6, oxygen_guage 7, diver_guage 8, diver 9), as in its Exp 5 prompt
CHANNEL_MAP = [(0, 1), (5, 6, 4), (5, 6, 9), (2,), (3, 1), (7, 8)]

GROUP = 8                                   # episodes per instance
VAL = [tuple(range(20_000_000 + GROUP * i, 20_000_000 + GROUP * (i + 1))) for i in range(16)]  # 128 episodes
TEST = list(range(10_000_000, 10_001_000))  # 1,000 episodes, never shown to the search
FRAMES, TRACES = 5, 2                       # feedback: last 5 steps of the 2 lowest-scoring episodes


def train_groups(run_seed, n=200):
    base = 30_000_000 + 1_000_000 * run_seed
    return [tuple(range(base + GROUP * i, base + GROUP * (i + 1))) for i in range(n)]


# --- the game ------------------------------------------------------------------------
def load_npz(path):
    d = np.load(path)
    keys = ["dense1/kernel", "dense1/bias", "dense2/kernel", "dense2/bias", "logits/kernel", "logits/bias"]
    return {n: d[f"actor/{k}"].astype(np.float32) for n, k in zip(SHAPES, keys)}


def observe(raw):
    return np.stack([raw[:, :, list(c)].any(axis=2) for c in CHANNEL_MAP], axis=2).reshape(-1).astype(np.float32)


def policy(net, x):
    h = np.tanh(np.tanh(x @ net["W1"] + net["b1"]) @ net["W2"] + net["b2"])
    z = h @ net["W3"] + net["b3"]
    p = np.exp(z - z.max())
    return p / p.sum()


def _end_cause(g):
    if g.oxygen <= 0:
        return "oxygen ran out"
    if g.sub_y == 0 and g.diver_count == 0 and not any(x[0:2] == [g.sub_x, g.sub_y] for x in g.e_fish + g.e_subs + g.e_bullets):
        return "surfaced with no diver"
    at = [g.sub_x, g.sub_y]
    for name, group in (("enemy fish", g.e_fish), ("enemy sub", g.e_subs), ("enemy bullet", g.e_bullets)):
        if any(x[0:2] == at for x in group):
            return f"hit by {name}"
    return "hit (enemy moved through the submarine)"


def run_episode(net, seed, keep_frames=FRAMES):
    env = Environment("seaquest", sticky_action_prob=0.1)
    env.seed(seed)
    env.reset()
    g = env.env
    rng = np.random.RandomState(seed + 1_000_003)
    ret, bonus, divers, surfaces, full_surfaces, steps = 0, 0, 0, 0, 0, 0
    acts, rows, frames, done = np.zeros(6, int), np.zeros(10, int), [], False
    while not done and steps < MAX_STEPS:
        raw = env.state()
        p = policy(net, observe(raw))
        a = int(min(np.searchsorted(np.cumsum(p), rng.random_sample()), 5))
        dc, surf, ox = g.diver_count, g.surface, g.oxygen
        r, done = env.act(a)
        executed = env.last_action
        acts[executed] += 1
        rows[g.sub_y] += 1
        if g.surface and not surf:
            surfaces += 1
            if dc == 6:
                full_surfaces += 1
                bonus += ox * 10 // 200
        if g.diver_count > dc:
            divers += g.diver_count - dc
        ret += r
        steps += 1
        if keep_frames:
            frames = (frames + [(raw, [round(float(x), 3) for x in p], a, executed, steps)])[-keep_frames:]
    return {"seed": seed, "return": ret, "length": steps, "kills": ret - bonus, "bonus": bonus, "divers": divers,
            "surfaces": surfaces, "full_surfaces": full_surfaces, "cause": _end_cause(g) if done else "step limit",
            "actions": acts.tolist(), "rows": rows.tolist(), "frames": frames, "final": env.state()}


_POOL = None


def _job(args):
    return run_episode(*args)


def run_episodes(net, seeds, keep_frames=FRAMES):
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(int(os.environ.get("WG_PROCS", max(1, (os.cpu_count() or 2) - 2))))
    return list(_POOL.map(_job, [(net, s, keep_frames) for s in seeds], chunksize=max(1, len(seeds) // 64)))


# --- feedback text -------------------------------------------------------------------
GLYPH = [(0, "S"), (1, "s"), (2, "*"), (5, "F"), (6, "U"), (4, "o"), (9, "D"), (3, "-"), (7, "O"), (8, "d")]


def render(raw):
    lines = []
    for r in range(10):
        row = [next((sym for k, sym in GLYPH if raw[r, c, k]), ".") for c in range(10)]
        lines.append(f"  row {r}  " + " ".join(row))
    return "\n".join(lines)


def summarize(eps, n_traces=TRACES):
    """Aggregates over the episodes, then the last frames of the lowest-scoring ones."""
    R = np.array([e["return"] for e in eps], float)
    n = len(eps)
    se = R.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    causes = {}
    for e in eps:
        causes[e["cause"]] = causes.get(e["cause"], 0) + 1
    acts, rows = np.sum([e["actions"] for e in eps], axis=0), np.sum([e["rows"] for e in eps], axis=0)
    m = lambda k: np.mean([e[k] for e in eps])  # noqa: E731
    out = [f"{n} episodes. Score per episode: mean {R.mean():.1f} (SE {se:.1f}), median {np.median(R):.0f}, "
           f"min {R.min():.0f}, max {R.max():.0f}. Length: mean {m('length'):.0f} steps.",
           "How episodes ended: " + ", ".join(f"{k} {v}" for k, v in sorted(causes.items(), key=lambda kv: -kv[1])) + ".",
           f"Per episode (mean): enemies shot {m('kills'):.1f}, divers picked up {m('divers'):.1f}, surfacings "
           f"{m('surfaces'):.1f} (with 6 divers {m('full_surfaces'):.2f}), surfacing bonus {m('bonus'):.1f}.",
           "Executed actions: " + ", ".join(f"{a} {100 * x / acts.sum():.0f}%" for a, x in zip(ACTIONS, acts)) + ".",
           "Time spent per submarine row (0 top .. 8): " + " ".join(f"{100 * x / rows.sum():.0f}%" for x in rows[:9]) + "."]
    for i in np.argsort(R, kind="stable")[:n_traces]:
        e = eps[i]
        out.append(f"\nEpisode seed {e['seed']}: score {e['return']}, {e['length']} steps, ended: {e['cause']}. "
                   f"Last {len(e['frames'])} steps (the grid the network saw, its action probabilities "
                   f"[{', '.join(ACTIONS)}], the sampled and the executed action):")
        for raw, p, a, ex, t in e["frames"]:
            out.append(f" step {t}: p={list(p)} sampled {ACTIONS[a]}, executed {ACTIONS[ex]}\n{render(raw)}")
        out.append(f" final state:\n{render(e['final'])}")
    return "\n".join(out)


# --- what EditGEPA needs --------------------------------------------------------------
def evaluate(weights, groups, with_feedback):
    """One score per group (its mean return) and, on request, the feedback text for all the episodes."""
    eps = run_episodes(weights, [s for g in groups for s in g], keep_frames=FRAMES if with_feedback else 0)
    scores = [float(np.mean([e["return"] for e in eps[i * GROUP:(i + 1) * GROUP]])) for i in range(len(groups))]
    return scores, summarize(eps) if with_feedback else None


def test(weights):
    R = np.array([e["return"] for e in run_episodes(weights, TEST, keep_frames=0)], float)
    return {"test_mean": float(R.mean()), "test_se": float(R.std(ddof=1) / np.sqrt(len(R))), "test_episodes": len(R)}


def seed_program():
    return language.last_block((DATA / "seaquest_hand_answer.md").read_text()).strip()


TASK = ("YOUR TASK: improve the weights of a policy network that plays this game, so that it scores as high as "
        "possible. There is no training: no data, no loss, no gradients, no fine-tuning; you choose every number "
        "yourself, by writing edits in the language described at the end. You start from a network that was written "
        "by hand from scratch, without feedback; its author's description and its full edits program are given. "
        "Networks you write are tested by playing episodes, and you get back what happened.\n\n")

PROTOCOL = (  # the pilot's wording of dspy_wsp.edit_gepa.PROTOCOL
    "\n\nHOW THIS WORKS: you are shown one network (its full edits program, applied to all-zero weights) and how it "
    "did on some test episodes. Reply with a brief explanation of what you change and why, then exactly one edits "
    "block. It is applied on top of the network shown; make its first line `# base: zero` to start from all-zero "
    "weights instead.")


def task():
    """The writer's system prompt: our task text, then the paper's Exp 5 prompt from THE GAME on (rules, source, input,
    network, edits language)."""
    text = (DATA / "seaquest_exp5_prompt.txt").read_text()
    return TASK + text[text.index("THE GAME:"):]


def context():
    """Shown before every parent: the hand-written seed network's design notes."""
    design = (DATA / "seaquest_hand_answer.md").read_text().split("```edits")[0].strip()
    return ("The original hand-written network's author described it as follows (the network below may have been "
            f"edited since):\n{design}")
