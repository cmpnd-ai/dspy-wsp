# Editing d1-omni-600M's weights to improve Typed Decisions

This folder documents the first `WeightSpace` run on a decision model. The writer was `openai/gpt-6.1-sol`. The student was Liquid AI's d1-omni-600M, run with MLX on a MacBook Pro with an M2 Max. A DSPy program called the model through DSPy's TypeSafe client and a local System One server.

The writer changed one tensor: the 1,024 weights of `head.scorer.3.weight`, the last layer of d1's option scorer. It computed the change in closed form from the model's own head features, with no gradients. On the 400 Typed Decisions test cases, which the writer never saw, the program's accuracy rose from 53.4% to 60.6%. The paired gain was +7.25 points with a standard error of 1.05 points. The edit improved 150 cases and made 63 worse.

The whole run took about 75 minutes on the laptop. The writer's session took 67 minutes, with 60 turns, 33 Python runs and 25 forward passes. The final check took about 5 minutes.

## The task

Typed Decisions (`LocalLLaMA/typed-decisions`, Apache-2.0) gives a model one record from one of four workflows and asks five typed questions about it. The workflows are agent traces, customer service messages, invoices and security incidents. The questions mix Choice, Noul (yes or no) and Score (ordered levels). Every decision has a gold probability distribution.

- The train split has 1,200 cases. The harness held out 300 of them for validation, 75 from each workflow, and gave the writer the other 900 as practice cases.
- The test split has 400 cases, 100 from each workflow. Only the final check used them.
- The metric follows the dataset's scorer (`bench/score_with_jobs.py`). A decision is right when the most likely predicted option is the gold distribution's most likely option. A case scores the share of its five decisions that are right. The writer also saw, for each decision, the predicted and gold options and the case's Brier score.

## Setup

- The hardware was an Apple M2 Max with 64 GB of memory, on macOS 15.7.7.
- The software was MLX 0.32.4, mlx-lm 0.32.2, PyTorch 2.15.0, DSPy 3.4.1, typesafe-sdk 0.7.5 and tokenizers 0.23.3.
- The code was commit `b0268a6` of this repository.
- The student was `LiquidAI/d1-omni-600M` at revision `02b55d7076f15129e59ab3f94783f32c4b088674`, text path only, in float32. Its text network has about 350M parameters: a 16-block bidirectional LFM2 encoder and a two-layer decision head.
- We loaded the model with our own MLX code, `D1Backend`. Before the run, we checked it against Liquid AI's reference code on 100 Typed Decisions questions. The largest probability difference was 6.9e-6.
- The program, `examples/typed_decisions.py`, has one `dspy.Predict` per workflow. Each signature has a `state` input and one `Choice`, `Noul` or `Score` output per question. DSPy sends each call through `dspy.experimental.TypeSafe` to a local `/v1/systemone` server that the backend runs.
- The writer had 25 forward passes. A `decide` pass could run up to 900 practice cases, or 450 per model when comparing both models. A `read` pass could read up to 1,024 questions. A validation run scored the whole validation set and returned only aggregate results.
- The writer's code ran without the bubblewrap sandbox, because bubblewrap does not run on macOS. It ran with the user's permissions.

The command, run from `examples/`:

```bash
OPENAI_API_KEY=... python typed_decisions.py --model d1 --framework mlx
```

## Baselines

We measured the unedited model on the 400 test cases two ways. The files are in `checks/`.

| Request format | Accuracy | KL from gold | Brier | ECE |
|---|---|---|---|---|
| The dataset scorer's protocol: each case's own state and questions, sent straight to the server | 57.5% | 0.352 | 0.173 | 0.064 |
| The DSPy program | 53.4% | 0.370 | 0.192 | 0.085 |

The DSPy adapter wraps the state with the signature's instructions and field descriptions, and d1-omni-600M does about 4 points worse on that format. The run optimized the DSPy program, so the program's 53.4% is the baseline for the run's result.

For comparison, the dataset's leaderboard lists Liquid AI's d1-2B at 63.7% and d1-3B at 65.6% under the scorer's protocol. It does not list d1-omni-600M.

## What the writer did

The writer's notes and turns are in `run/`. Its passes fell into four stages.

1. It ran the original model on all 900 practice cases (56.6%). It then read the decision head's features for every practice question in five read passes of 900 questions each.
2. It split its practice cases into two halves of 450 at random. On the first half it fit a correction to the scorer's last layer with ridge regression, choosing the regularization strength by three-fold cross-validation within that half. The target was the gold distribution's log probabilities. It tested three strengths of the correction on the other half: half strength gained +4.6 ± 0.7 points, full strength +8.0 ± 1.1, and 1.5 times strength +2.6 ± 1.3. The first validation run confirmed the full-strength edit at +7.0 ± 1.3 points.
3. It refit the same correction on all 900 practice cases. A validation run measured +8.1 ± 1.3 points, from 57.5% to 65.6%.
4. It split the correction into parts and measured each one:
   - A part that only shifts each question's preference between options gained +0.1 ± 1.3 points on validation.
   - The remaining part, which depends on the record's content, gained +2.7 ± 1.0.
   - The full correction gained +8.1 ± 1.3, more than the two parts alone.
   - The 512 largest coefficients alone, and the other 512 alone, each gained less than half a point on practice cases.

   The writer concluded that the gain comes from a distributed change to how the scorer reads its features, not from a few weights or from shifting option frequencies.

The writer submitted the full-strength correction fit on all 900 practice cases. Its last validation run measured that exact edit again at +8.1 ± 1.3 points.

The writer looked at validation results five times. Three runs measured full-strength corrections: the one fit on half the practice cases once, and the submitted one twice. Two runs measured parts of the submitted correction. The test split stayed untouched, so the final check below carries no selection.

## Results

On the 400 test cases, through the DSPy program:

| | Accuracy | Standard error |
|---|---|---|
| Original weights | 53.35% | 1.07 |
| Edited weights | 60.60% | 1.18 |
| Paired gain | +7.25 points | 1.05 |

The edit improved 150 cases and made 63 worse. The model is deterministic, so the paired comparison has no sampling noise. The standard error comes only from which cases are in the test split.

The gain differs by workflow:

| Workflow | Paired gain on test |
|---|---|
| Agent traces | +12.8 ± 2.6 points |
| Customer service | +7.8 ± 2.1 points |
| Security incidents | +5.0 ± 1.4 points |
| Invoices | +3.4 ± 1.9 points |

The largest gains were on questions the original model often got wrong. Invoice urgency rose from 34% to 60%. Agent trace action rose from 38% to 56%. Agent trace review rose from 55% to 73%. Customer churn risk rose from 56% to 72%. Three invoice questions got slightly worse: disposition fell from 40% to 36%, discrepancy severity from 47% to 43%, and order matching from 62% to 61%.

Under the dataset scorer's protocol, the edited model also improved on every measure. Its distributions moved much closer to the gold distributions:

| Scorer protocol, 400 test cases | Accuracy | KL from gold | Brier | ECE |
|---|---|---|---|---|
| Original weights | 57.5% | 0.352 | 0.173 | 0.064 |
| Edited weights | 60.6% | 0.233 | 0.132 | 0.038 |

The writer fit the edit on DSPy's request format, so the accuracy gain under the scorer's protocol is smaller: +3.1 points instead of +7.25.

## What the edit is

The edit adds a computed vector to the 1,024 weights that turn the scorer's hidden layer into one logit per option. The writer derived it by ridge regression on head features it read through the forward-pass tool, which the decision-model rules allow. It is a closed-form fit of the scorer's last layer to practice data. It is not a mechanism the writer found and repaired by hand, and it is close to fitting a linear probe on features the model already computes. The validation and test results show that it carries over to records the writer never saw.

## The smoke run

Before the full run, a three-pass smoke run used 250 training cases, 50 validation cases and 40 test cases. Its report and final answer are in `checks/`. The writer fit a similar correction to the same tensor on 51 practice cases. It gained +6.3 ± 3.5 points on practice but lost 8.0 ± 3.0 points on validation, so the writer removed it and submitted the original weights. In the full run, the larger fitting set and the cross-validated regularization made the difference.

## Files

- `run/report.json` has the budget used, every validation run, the final check and the patch summary.
- `run/final_answer.md` is the writer's final answer.
- `run/writer_notes.md` holds the notes the writer kept during the session.
- `run/task_prompt.txt` is the task the writer was given.
- `run/transcript.jsonl` lists every Python run and forward pass with its output.
- `run/turns/` holds each turn's prompt and the writer's reply.
- `run/patch/` is the submitted patch. `Patch.load` rebuilds the edited weights from the pinned d1-omni-600M and checks their hashes.
- `run/final_check_scores.jsonl` has one line per test case with both scores and the metric's feedback for both.
- `checks/` holds the baselines, the edited model under the scorer's protocol, and the smoke run's report and final answer.

## How to reproduce

```bash
pip install -e ".[examples]"
cd examples
python typed_decisions.py --model d1 --framework mlx --baseline protocol
python typed_decisions.py --model d1 --framework mlx --baseline program
OPENAI_API_KEY=... python typed_decisions.py --model d1 --framework mlx --smoke
OPENAI_API_KEY=... python typed_decisions.py --model d1 --framework mlx
```

The writer's answers are sampled, so a new run will make different choices. d1-omni-600M uses the LFM Open License v1.0, which limits commercial use for organizations with $10 million or more in annual revenue. Check it before you share edited weights.
