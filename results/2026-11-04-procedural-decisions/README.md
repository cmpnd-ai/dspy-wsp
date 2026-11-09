# Weight edits on procedural decision problems: d1-omni-600M and NanoJev

This folder documents `WeightSpace` runs on three tasks from Procedural Typed Decisions, for d1-omni-600M and NanoJev, and compares each edit with `dspy.experimental.ReAnchor`. The writer was `openai/gpt-6.1-sol`. Everything ran overnight on a MacBook Pro with an M2 Max, with MLX, through `examples/overnight_procedural.sh`.

In both runs the writer changed one tensor of the decision head. On the 400 test cases, the program's accuracy rose:

| Model | Tensor edited | Original | Edited | Paired gain | Improved / worse |
|---|---|---|---|---|---|
| d1-omni-600M | `head.scorer.3.weight` (1,024 weights) | 52.3% | 72.3% | +20.0 ± 2.3 points | 179 / 54 |
| NanoJev | `scalar.weight` (1,024 weights) | 49.3% | 72.3% | +23.1 ± 2.3 points | 193 / 54 |

These tasks have balanced labels. A control that only shifted each question's preference between options gained nothing for NanoJev and lost 10 points for d1, so the gains come from how the head reads each record.

## The task

Procedural Typed Decisions (`tasksource/procedural-typed-decisions`, Apache-2.0) generates decision problems from rules that each record states. The labels are exact, and no answer exceeds 60% of a question's answers. We used the three tasks whose questions and options are the same in every record, because ReAnchor fits settings per question and option:

| Task | Questions |
|---|---|
| Event-state reconstruction | the ticket's current owner (Choice of 4), whether it is open (Noul), its current severity (Score of 3), from a shuffled event log with voided entries |
| Multi-view adjudication | the request's intent (Choice of 4), whether it is urgent (Noul), how much it blocks work (Score of 3), from signals and stated rules |
| Partial-observation calibration | whether an incident is real (Noul; the gold answer is an exact Bayesian posterior), from up to 6 sensor reports |

- Practice and validation cases came from the dataset's validation split: 400 per task, 1,200 in all. WeightSpace held out 300 of them for validation, 100 per task, and gave the writer the other 900.
- The 400 test cases came from the dataset's test split: 134, 133 and 133 per task.
- The 400 test cases hold 934 decisions. Always choosing each question's most common answer gets 41.5% of them right.
- The metric is the one used for Typed Decisions: a decision is right when the most likely predicted option is the gold distribution's most likely option. The program's score for a case is the share of its decisions that are right, and the run's scores average those shares over cases. Partial-observation cases have one question and the others three, so these case averages differ slightly from the per-decision accuracies in the baseline tables.

## Setup

- The models, loaders, harness and settings match the Typed Decisions runs in `../2026-11-04-typed-decisions-d1/` and `../2026-11-04-typed-decisions-nanojev/`: 25 passes, up to 900 items per decide pass and 1,024 questions per read pass, and a 300-case validation holdout.
- The code was commit `e09f381` of this repository. The driver ran four steps per model: two baselines, the WeightSpace run and the ReAnchor comparison. A watchdog would have stopped any step above 20 GB of memory. None came close.
- NanoJev's reference code requires a description for every Choice option. The event-state owner question has none, so our NanoJev loader reads such an option as its key.

```bash
examples/overnight_procedural.sh d1 nanojev
```

| Step | d1-omni-600M | NanoJev |
|---|---|---|
| Baselines | 2 minutes | 5 minutes |
| WeightSpace (writer's session) | 60 minutes (57) | 117 minutes (111) |
| ReAnchor comparison | 12 minutes | 26 minutes |

## Baselines

The unedited models on the 400 test cases:

| Model | Request format | Accuracy | KL from gold | Brier | ECE |
|---|---|---|---|---|---|
| d1-omni-600M | the dataset's own request | 55.0% | 1.097 | 0.609 | 0.220 |
| d1-omni-600M | the DSPy program | 52.2% | 1.168 | 0.616 | 0.207 |
| NanoJev | the dataset's own request | 48.9% | 1.036 | 0.625 | 0.179 |
| NanoJev | the DSPy program | 49.8% | 1.024 | 0.617 | 0.148 |

## What the writers did

Both sessions followed the same steps as on Typed Decisions. Each read the head's features through read passes, fit a correction to the head's readout by ridge regression on practice cases, and tested strengths on practice cases held out from the fit. Each then confirmed the correction on validation and spent the remaining passes on checks of how it works. The notes and turns are in `d1/run/` and `nanojev/run/`.

- **d1** compared three versions of the correction on validation: +13.9, +14.7 and +18.7 points. It submitted the third, fit on 840 practice cases. Its checks found that:
  - the prior-only part of the edit lost 10.3 points on practice cases;
  - keeping only the edit's 64, 256 or 512 largest coefficients recovered less than the full edit;
  - reversing the edit cost 1.0 point.
- **NanoJev** compared three versions on validation: +19.1, +23.6 and +17.7 points. It submitted the second, fit on all 900 practice cases with the regularization chosen by three-fold cross-validation on practice cases. Its prior-only control gained +0.2 ± 2.9 points, and reversing the edit reversed the model's response to several decisive cues.

Each writer chose among three versions using validation results, so their validation gains carry some selection. The test cases were untouched.

## Results

On the 400 test cases, through the DSPy program:

| Task | d1-omni-600M gain | NanoJev gain |
|---|---|---|
| Event-state reconstruction | +10.0 ± 2.8 points | +13.9 ± 2.6 points |
| Multi-view adjudication | +21.6 ± 3.2 points | +15.5 ± 3.2 points |
| Partial-observation calibration | +28.6 ± 5.3 points | +39.8 ± 5.3 points |
| All 400 cases | +20.0 ± 2.3 points | +23.1 ± 2.3 points |

Under the dataset's own request format:

| Model | Accuracy | KL from gold | Brier | ECE |
|---|---|---|---|---|
| d1-omni-600M, original | 55.0% | 1.097 | 0.609 | 0.220 |
| d1-omni-600M, edited | 55.8% | 0.884 | 0.541 | 0.106 |
| NanoJev, original | 48.9% | 1.036 | 0.625 | 0.179 |
| NanoJev, edited | 66.5% | 0.684 | 0.410 | 0.042 |

NanoJev's edit carries over to the dataset's request format. d1's edit improves its distributions under that format but barely changes its accuracy (+0.8 points). The writer fit d1's edit on the DSPy program's requests, which wrap the record with the signature's instructions and field descriptions. d1 reads the two formats differently enough that a readout fit to one format does not move accuracy on the other.

## With ReAnchor

ReAnchor fits how the program turns probabilities into answers: a threshold for every Noul, cut points for every Score and a weight for every Choice option. The comparison scores each version's decided answers on the 400 test cases. ReAnchor fit on the WeightSpace run's 900 practice cases. See `../2026-11-04-typed-decisions-nanojev/reanchor/` for the metric and method.

| Version | d1-omni-600M | NanoJev |
|---|---|---|
| Plain | 52.3% | 47.8% |
| ReAnchor | 58.7% (+6.4 ± 2.2) | 52.8% (+5.0 ± 1.7) |
| WeightSpace | 64.2% (+11.9 ± 2.3) | 68.4% (+20.6 ± 2.3) |
| WeightSpace + ReAnchor | 69.9% (+17.7 ± 2.4) | 71.3% (+23.4 ± 2.3) |

Gains are over plain. ReAnchor added to WeightSpace gained +5.8 ± 1.0 points for d1 and +2.8 ± 1.2 for NanoJev.

- ReAnchor alone helps less here than on Typed Decisions. The labels are balanced, so there are few skewed defaults for its weights and thresholds to correct.
- Score questions show a gap between the two metrics. With the edits, the most likely Score level is right 74.9% of the time for d1 and 78.3% for NanoJev. DSPy's default decoding rounds the probability-weighted average level instead, and that returns the right level only 38.2% and 60.7% of the time. ReAnchor's cut points recover part of the gap: d1's Score accuracy rises to 63.3% and NanoJev's to 70.8%. This is why WeightSpace's decided accuracy sits below its probability-based accuracy of 72.3% for both models.

## What this run shows

- On balanced, rule-based tasks, a closed-form edit to one head readout raised both models' test accuracy by 20 to 23 points, and prior-only versions of the edits gave no gain. The edits change how the head reads features the backbone already computes.
- As in the Typed Decisions runs, the edits are fits of a readout to practice data, close to fitting a linear probe on frozen features. The validation and test results show that they generalize to new records from the same tasks.
- d1's edit is specific to the request format it was fit on.
- ReAnchor is a cheap complement. On top of an edit, its main effect is to set Score cut points that match the edited probabilities.

## Files

For each model, `d1/` and `nanojev/` hold:

- `run/report.json`, `run/final_answer.md`, `run/writer_notes.md`, `run/task_prompt.txt`, `run/transcript.jsonl` and `run/turns/`: the WeightSpace run.
- `run/patch/`: the submitted patch.
- `run/final_check_scores.jsonl`: both scores and the metric's feedback for every test case.
- `checks/`: the two baselines and the edited model under the dataset's request format.
- `reanchor/`: the comparison and both ReAnchor reports.

`overnight.log` records when each step started and ended.

## How to reproduce

```bash
pip install -e ".[examples]"
OPENAI_API_KEY=... examples/overnight_procedural.sh d1 nanojev
```

Or one step at a time, from `examples/`:

```bash
python procedural_decisions.py --model d1 --step baseline-protocol
python procedural_decisions.py --model d1 --step baseline-program
OPENAI_API_KEY=... python procedural_decisions.py --model d1 --step optimize --writer sol
python procedural_decisions.py --model d1 --step compare --writer sol
```

The writer's answers are sampled, so a new run will make different choices.
