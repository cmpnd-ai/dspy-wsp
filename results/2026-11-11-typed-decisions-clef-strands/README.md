# Typed Decisions on Clef-Flash and Strands Decider 2B

This folder documents two `WeightSpace` runs on the Typed Decisions benchmark with `openai/gpt-6-astra` at maximum reasoning effort as the writer. Each run edited one decision model in 25 forward passes with 300 validation cases. We then scored every edit with and without `dspy.experimental.ReAnchor` on the 400 test cases, which the writer never saw. Both runs used MLX on a MacBook Pro.

The settings match the runs in [`../2026-11-07-typed-decisions-astra/`](../2026-11-07-typed-decisions-astra/README.md), which cover d1-omni-600M, NanoJev and d1-2B.

## Results

Scored on the answers each program returns. Each gain is the paired gain over the plain model in points, with its standard error.

| Version | Strands Decider 2B | Clef-Flash (9B) |
|---|---|---|
| Plain | 53.0% | 68.8% |
| ReAnchor | 66.7% (+13.7 ± 1.4) | 70.3% (+1.5 ± 0.8) |
| WeightSpace | 63.4% (+10.4 ± 1.1) | 70.7% (+1.9 ± 0.7) |
| WeightSpace + ReAnchor | 73.0% (+20.0 ± 1.3) | 75.3% (+6.5 ± 0.9) |

ReAnchor adds 9.6 ± 0.9 points to the Strands edit and 4.7 ± 0.7 points to the Clef-Flash edit.

## By question type

Accuracy of the returned answers, by question type. The test set holds 600 Choice, 600 Noul and 800 Score decisions.

| Strands Decider 2B | Choice | Noul | Score |
|---|---|---|---|
| Plain | 60.5% | 71.2% | 33.6% |
| ReAnchor | 64.8% | 75.3% | 61.5% |
| WeightSpace | 70.2% | 81.0% | 45.0% |
| WeightSpace + ReAnchor | 71.3% | 81.7% | 67.6% |

| Clef-Flash | Choice | Noul | Score |
|---|---|---|---|
| Plain | 68.5% | 81.0% | 59.9% |
| ReAnchor | 68.0% | 79.8% | 64.8% |
| WeightSpace | 73.0% | 84.7% | 58.4% |
| WeightSpace + ReAnchor | 70.8% | 84.3% | 71.9% |

Score questions carry most of ReAnchor's gain on both models. The plain Strands program answers 33.6% of them correctly, and its most likely level is correct on 50.6%. ReAnchor refits the cut points that turn a Score distribution into a level.

## Two measures of accuracy

Each run's `report.json` scores the option with the highest probability. The comparisons in `reanchor/` score the answer the program returns, which is the measure in the tables above.

| Run | Original | Edited | Paired gain on the most likely option |
|---|---|---|---|
| Strands Decider 2B | 59.8% | 72.7% | +13.0 ± 1.2 |
| Clef-Flash | 69.7% | 75.5% | +5.8 ± 0.8 |

The benchmark's public leaderboard lists Clef-Flash at 71.0% on the most likely option. It does not list Strands Decider 2B.

## The edits

| Run | Tensors changed | Turns | Python runs | Time | Writer cost |
|---|---|---|---|---|---|
| Strands Decider 2B | `k.weight` (all 1,572,864 entries, a rank-one change) | 69 | 41 | 68 min | $6.80 |
| Clef-Flash | `residual_scorer.0.weight` (98,302 of 4,194,304 entries), `residual_scorer.0.bias` (24), `residual_scorer.3.weight` (24) | 95 | 62 | 154 min | $10.78 |

The time runs from the first turn to the saved patch. The writer cost comes from each run's per-turn usage records.

**Strands Decider 2B.** The backend opens the whole network to the writer: the backbone, with the LoRA adapter folded in, and the pointer head. The writer edited only the head. It added a rank-one term to the key projection `k.weight`, derived by closed-form ridge regression on 540 practice cases. Its notes describe a readout bottleneck: ten dimensions hold about 90% of the option activations' energy, and the repair's benefit came from the other dimensions.

**Clef-Flash.** The backend keeps the backbone fixed, so the writer could edit only the joint schema head. It repurposed 24 low-contribution neurons of the residual scorer. Two give a shared linear correction, and 22 give gated corrections for urgency and for customer action and review decisions.

## Files

- `<model>/run/` holds each run: the report, the task prompt, every turn, the transcript, the writer's notes, its final answer, the per-case final check scores and the patch.
- `<model>/reanchor/` holds the four-way comparison with ReAnchor.
- The Strands patch applies to the folded checkpoint that `StrandsBackend` writes, and its manifest names the released checkpoint and revision.

## Reproduce

From `examples/`, with dspy-wsp 0.2.0:

```bash
OPENAI_API_KEY=... python typed_decisions.py --model strands-2b --framework mlx --writer astra
python typed_decisions_reanchor.py --model strands-2b --framework mlx --wsp-run runs/typed-decisions/strands-2b-astra --out runs/typed-decisions/reanchor_strands-2b-astra
```

Use `--model clef-flash` for Clef-Flash, which needs about 20 GB of memory.
