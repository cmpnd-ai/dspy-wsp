# Typed Decisions with `gpt-6-astra` as the writer

This folder documents three `WeightSpace` runs on the Typed Decisions benchmark with `openai/gpt-6-astra` at maximum reasoning effort as the writer. Each run edited one decision model in one conversation with 25 forward passes and 300 validation cases. We then scored every edit with and without `dspy.experimental.ReAnchor` on the 400 test cases, which the writer never saw. All runs used MLX on a MacBook Pro.

## Results

Scored on the answers each program returns. Each gain is the paired gain over the plain model in points, with its standard error.

| Version | d1-omni-600M | NanoJev (Qwen3-0.6B) | d1-2B |
|---|---|---|---|
| Plain | 50.9% | 39.2% | 60.6% |
| ReAnchor | 60.2% (+9.3 ± 1.1) | 50.7% (+11.6 ± 1.3) | 70.1% (+9.5 ± 1.2) |
| WeightSpace | 66.9% (+16.1 ± 1.2) | 52.4% (+13.3 ± 1.1) | 74.0% (+13.4 ± 1.2) |
| WeightSpace + ReAnchor | 69.1% (+18.3 ± 1.3) | 56.4% (+17.3 ± 1.3) | 77.9% (+17.3 ± 1.3) |

The same comparison for the `openai/gpt-6.1-sol` edits, with the same settings:

| Version | d1-omni-600M | NanoJev (Qwen3-0.6B) | d1-2B |
|---|---|---|---|
| WeightSpace | 56.7% (+5.8 ± 1.0) | 51.3% (+12.1 ± 1.2) | 60.0% (−0.6 ± 1.3) |
| WeightSpace + ReAnchor | 62.9% (+12.1 ± 1.2) | 55.7% (+16.6 ± 1.4) | 70.5% (+9.9 ± 1.3) |

## The paper's rule

In the runs above, the writer could recompute the decision head in its kernel and fit edits there. The paper's rule does not allow this: its writer could run the network only through the forward-pass tool. `d1-omni-600m-strict/` holds a run of `gpt-6-astra` on d1-omni-600M under the paper's rule (`simulate_head=False`, `--strict` in the example script), with the same settings otherwise.

| d1-omni-600M, `gpt-6-astra` | Head recomputed in the kernel | Paper's rule |
|---|---|---|
| Plain | 50.9% | 50.9% |
| ReAnchor | 60.2% (+9.3 ± 1.1) | 60.2% (+9.3 ± 1.1) |
| WeightSpace | 66.9% (+16.1 ± 1.2) | 53.7% (+2.8 ± 0.7) |
| WeightSpace + ReAnchor | 69.1% (+18.3 ± 1.3) | 60.2% (+9.3 ± 1.2) |

Under the paper's rule, the writer reached the same diagnosis: the scorer underused cues that were already in its inputs. It edited the same scorer tensors, `head.scorer.1` and `head.scorer.3`, and changed 12 scorer neurons to read payment deadline, constraint violation and security cues. Every check of an edit cost a forward pass, so it repaired two groups of cues in 25 passes. The run took 73 minutes and cost $9.25 in writer calls. It used 75 Python runs, and none of them computed the head's output on an input.

## Two measures of accuracy

Each run's `report.json` scores the option with the highest probability. The comparisons in `reanchor/` score the answer the program returns, which is the measure in the tables above. The two measures differ when the program's thresholds or cut points pick a different answer from the most likely option.

| Run | Original | Edited | Paired gain on the most likely option |
|---|---|---|---|
| d1-omni-600M, `gpt-6-astra` | 53.4% | 68.3% | +15.0 ± 1.2 |
| NanoJev, `gpt-6-astra` | 38.6% | 57.3% | +18.8 ± 1.3 |
| d1-2B, `gpt-6-astra` | 62.2% | 77.4% | +15.2 ± 1.2 |
| d1-2B, `gpt-6.1-sol` | 62.2% | 66.5% | +4.3 ± 1.2 |

The `gpt-6.1-sol` edit to d1-2B raised the most likely option's accuracy by 4.3 points. It did not change the accuracy of the program's answers.

## The edits

| Run | Tensors changed | Turns | Python runs | Time | Writer cost |
|---|---|---|---|---|---|
| d1-omni-600M, `gpt-6-astra` | `head.scorer.1.weight` (112,640 of 1,048,576 entries), `head.scorer.1.bias` (110), `head.scorer.3.weight` (110) | 74 | 43 | 78 min | $7.63 |
| NanoJev, `gpt-6-astra` | `scalar.weight` (1,024) | 68 | 41 | 167 min | $6.92 |
| d1-2B, `gpt-6-astra` | `layers.23.feed_forward.w2.weight` (all 15.7 million entries, a rank-9 change) | 65 | 38 | 83 min | $5.42 |
| d1-2B, `gpt-6.1-sol` | `layers.23.post_ffn_norm.weight` (384 of 2,048) | 59 | 33 | | |

d1-2B has no decision head, so both writers edited its backbone. The writer cost comes from each run's per-turn usage records. The `gpt-6.1-sol` run on d1-2B predates those records.

## Files

- `<model>/run/` holds each `gpt-6-astra` run: the report, the task prompt, every turn, the transcript, the writer's notes, its final answer, the per-case final check scores and the patch.
- `<model>/reanchor/` holds the four-way comparison with ReAnchor.
- `d1-omni-600m-strict/` holds the run under the paper's rule.
- `sol/` holds the `gpt-6.1-sol` comparisons and the `gpt-6.1-sol` run on d1-2B. The other `gpt-6.1-sol` runs are in `../2026-11-04-typed-decisions-d1/` and `../2026-11-04-typed-decisions-nanojev/`.
- The d1-2B patch has only its manifest, because the edit file is 25 MB. The manifest has the base and edited hashes of the changed tensor.

## Reproduce

From `examples/`, with the code at commit `3812088`:

```bash
OPENAI_API_KEY=... python typed_decisions.py --model d1 --framework mlx --writer astra
python typed_decisions_reanchor.py --model d1 --framework mlx --wsp-run runs/typed-decisions/d1-astra --out runs/typed-decisions/reanchor_d1-astra
```

Use `--model nanojev` or `--model d1-2b` for the other models.

The run under the paper's rule needs the code at commit `73e61ba` or later:

```bash
OPENAI_API_KEY=... python typed_decisions.py --model d1 --framework mlx --writer astra --strict
python typed_decisions_reanchor.py --model d1 --framework mlx --wsp-run runs/typed-decisions/d1-astra-strict --out runs/typed-decisions/reanchor_d1-astra-strict
```
