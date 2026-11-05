# EditGEPA on MinAtar Seaquest: pilot, seed 0 (2026-11-05)

GEPA population search over edits programs, starting from the network the paper's writer built by hand in Experiment 5 (53.1 in the paper). The writer was `gpt-6-astra` at `xhigh` effort on Azure. The budget was 6,400 episodes, the same as 50 passes of 128. The settings are the recipe's defaults in [examples/seaquest_gepa](../../examples/seaquest_gepa/).

This run used the harness that `dspy_wsp.edit_gepa` was later packaged from. The algorithm, prompts, seeds and GEPA settings are the same. The differences:
- the writer was called through the OpenAI client rather than `dspy.LM`;
- the log fields are named `cand`, `kind` and `used`, where the API now writes `program`, `split` and `units_used`;
- the message for a skipped statement is worded slightly differently.

## Results

All scores are on the same 1,000 test episodes (seeds 10,000,000–10,000,999), which the search never saw. Differences are paired by episode seed.

| Network | Test score | vs start (paired) |
|---|---|---|
| Hand-written network (start) | 53.3 ± 1.3 | |
| **EditGEPA best** (selected on validation) | **85.6 ± 1.1** | **+32.3 ± 1.8** |
| *Paper's PPO, 500M steps, 3 seeds (their evaluation)* | *7.5, 4.4, 88.9* | |

| Cost and time | |
|---|---|
| Episodes used | 6,272 of 6,400 |
| Writer calls | 37 (16.4K reasoning tokens per call on average) |
| Writer tokens | 503K in (177K cached), 656K out |
| Writer cost | $36.26 at $10 / $1 / $50 per M tokens |
| Search time | 5 h 21 min on a 2-vCPU VM, mostly waiting on the writer |
| Test time | 1.4 min |

Every network admitted to the population, scored on validation (128 episodes) and on test:

| Program | Admitted after (episodes) | Validation | Test |
|---|---|---|---|
| 8f740a3be2 (start) | 128 | 58.1 | 53.3 ± 1.3 |
| ddeb73e764 | 384 | 74.8 | 75.5 ± 1.3 |
| 7e70b5193b | 640 | 72.9 | 72.1 ± 1.1 |
| 5061244e8d | 1,024 | 77.0 | 78.0 ± 1.2 |
| **b8c0906a43** (selected) | 1,536 | **83.4** | **85.6 ± 1.1** |
| 1da8d769d6 | 2,816 | 71.0 | 71.4 ± 1.0 |
| aa9679f6fc | 3,840 | 77.6 | 74.1 ± 1.2 |
| 170ec36d2a | 4,096 | 82.7 | 81.8 ± 1.1 |
| 094735256e | 4,864 | 77.9 | 80.0 ± 1.2 |
| b21b6f1b5b | 5,248 | 56.2 | 65.6 ± 1.3 |
| 92c368c575 | 5,632 | 83.0 | 78.3 ± 1.3 |
| a7a42f73bd | 6,144 | 77.8 | 77.0 ± 1.3 |

## Findings

1. **The search raised the score by 61%.** The selected network differs from the start in 457 of 20,486 weights, spread over all six tensors: W1 331, W2 67, W3 29, b2 16, b1 8 and b3 6. Its episodes scoring under 20 fell from 31% to 9%.
2. **The best network came early.** It was admitted after 1,536 episodes and 7 writer calls ($6.61). The remaining 4,700 episodes and 30 writer calls ($29.65) admitted 7 more networks, none better on validation or on test. Selecting on validation picked the network that is also best on test.
3. **Most writer calls went to weaker parents.** The selected network was the parent in 10 of 37 calls. The start and the first three children were the parent in 23. With 16 validation instances of 8 episodes each, noise keeps many networks on the Pareto frontier. This is our reading of the counts, not a tested claim.
4. **Calls are expensive because the writer starts fresh each time.** It reasons over the parent's whole program on every call, about 16K reasoning tokens per call. In the same pilot, a single ongoing conversation with the same writer used about 1.6K reasoning tokens per call.

For context, the same pilot ran two other arms from the same start with the same 6,400-episode budget, from a separate harness that isn't part of this PR. A single conversation, the paper's Experiment 6 loop, reached **135.6 ± 1.6** for $10.43. Blind OpenAI-ES reached 52.2 ± 1.3, no gain. The single conversation's largest step came from fixing a bug in its own previous edit. That path isn't open to EditGEPA: a broken child is dropped, and the next call never learns it existed.

## Caveats

- One seed. Whether population search beats a single conversation, or the reverse, needs more seeds.
- The PPO numbers are the paper's evaluation on its own episodes. Our evaluator reproduces the paper's score for the start network (53.33 against 53.11), but the PPO comparison is unpaired.
- The writer saw the game's full source code in its prompt; PPO did not.

## Files

```
run/result.json         summary: best validation score, episodes, writer tokens and $, compile and test time
run/final_program.txt   the selected network's edits program
run/programs.jsonl      the 12 networks admitted, with full programs and validation scores
run/proposals.jsonl     all 37 writer calls: parent, the feedback it read, its reply, tokens, $ so far
run/evals.jsonl         every evaluation (minibatch and validation), with episodes used so far
```

Reproduce with `python run_gepa.py --seed 0` in `examples/seaquest_gepa` (about $36 of writer tokens and 5 h at `xhigh`). Writer replies vary between runs, so expect a different trajectory.
