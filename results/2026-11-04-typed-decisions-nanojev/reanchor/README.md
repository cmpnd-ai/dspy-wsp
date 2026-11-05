# NanoJev with ReAnchor, WeightSpace and both

This folder compares four versions of the NanoJev Typed Decisions program on the 400 test cases:

- **Plain** is the original program on the original weights.
- **ReAnchor** is the original weights with `dspy.experimental.ReAnchor` applied. ReAnchor changes no weights. It fits how each `Predict` turns probabilities into an answer: a threshold for every Noul, cut points for every Score and a weight for every Choice option.
- **WeightSpace** is the edited weights from `../` (one edit to `scalar.weight`), with DSPy's default decoding.
- **WeightSpace + ReAnchor** is the edited weights with ReAnchor fitted on top.

| Version | Decided accuracy | Paired gain over plain | Improved / worse |
|---|---|---|---|
| Plain | 39.2% ± 1.2 | | |
| ReAnchor | 50.7% ± 1.3 | +11.6 ± 1.3 points | 196 / 68 |
| WeightSpace | 51.3% ± 1.1 | +12.1 ± 1.2 points | 213 / 63 |
| WeightSpace + ReAnchor | 55.7% ± 1.2 | +16.6 ± 1.4 points | 238 / 56 |

Adding ReAnchor to the edited weights gained +4.5 ± 0.8 points over WeightSpace alone, improving 124 cases and making 52 worse.

## The metric

The metric scores the answer the DSPy program returns. A decision is right when the program's decided answer is the gold distribution's most likely option: the chosen option for a Choice, true or false for a Noul, and the level for a Score.

This differs from the probability-based accuracy used in `../`. That metric takes the most likely option straight from the model's probabilities, and ReAnchor cannot change it. The two metrics agree for Choice and Noul questions under DSPy's default decoding. They can disagree for Score questions, because by default DSPy picks a Score's level by rounding the probability-weighted average level, not by taking the most likely level.

| Version | Decided accuracy | Probability accuracy |
|---|---|---|
| Plain | 39.2% | 38.6% |
| ReAnchor | 50.7% | 38.6% |
| WeightSpace | 51.3% | 55.7% |
| WeightSpace + ReAnchor | 55.7% | 55.7% |

## Setup

- ReAnchor fitted on the 900 practice cases of the WeightSpace run and reported on its 300 validation cases. The script read the split from the run's practice file. Nothing was fitted on the test cases.
- ReAnchor keeps a fitted setting only when it beats the original setting on held-out folds of the training cases. Otherwise it restores the original.
- The program used DSPy's TypeSafe client with caching on. The cache key includes a hash of the current weights, so cached answers always match the weights that produced them.
- The script is `examples/typed_decisions_reanchor.py`, run from `examples/` at commit `7379196`:

```bash
python typed_decisions_reanchor.py --model nanojev --framework mlx --wsp-run runs/typed-decisions/nanojev
```

The whole comparison took 62 minutes on the M2 Max. Each ReAnchor fit took about 25 minutes. Most of that time went to answering the 900 practice cases once; the search over settings reused cached answers.

## What each method changed

Decided accuracy by question type on the test cases:

| Version | Choice (600 decisions) | Noul (600) | Score (800) |
|---|---|---|---|
| Plain | 38.8% | 49.3% | 31.8% |
| ReAnchor | 46.0% | 67.7% | 41.5% |
| WeightSpace | 49.2% | 71.5% | 37.6% |
| WeightSpace + ReAnchor | 49.0% | 71.0% | 49.2% |

- ReAnchor on the original weights corrects NanoJev's skewed defaults at the decision step. It kept fitted settings for 12 of the 20 questions: weights for 4 of the 6 Choice questions, thresholds for 4 of the 6 Noul questions and cut points for 4 of the 8 Score questions. Some fitted values are large, e.g., a weight of 33.8 on "human review" for the agent trace action, which shows how far the original probabilities were from the right answers.
- WeightSpace improves the probabilities themselves. Its gains on Choice and Noul questions exceed ReAnchor's. On Score questions it reaches 48.6% by probability accuracy, but DSPy's default rounding returns the right level only 37.6% of the time.
- ReAnchor on top of WeightSpace kept new cut points for 7 of the 8 Score questions, and Score accuracy rose to 49.2%. It changed little for Choice and Noul questions, where the edit had already corrected the defaults.

The two methods overlap in what they fix. Both correct NanoJev's fixed preferences for some options, and on their own they reach about the same accuracy. In the WeightSpace run, the writer found that fixed per-option shifts alone took the practice score from 36.2% to 51.2%, close to what ReAnchor reaches. WeightSpace also improves how the head reads each record. ReAnchor then sets the decision rules for the edited probabilities. Together they gain 16.6 points.

## Files

- `comparison.json` has every version's accuracy by metric and question type, the paired comparisons and the run times.
- `reanchor_report.json` and `wsp_reanchor_report.json` are ReAnchor's reports: the training and validation scores before and after fitting, and every fitted or restored setting.
