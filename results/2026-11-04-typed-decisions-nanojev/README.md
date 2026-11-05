# Editing NanoJev's weights to improve Typed Decisions

This folder documents a `WeightSpace` run on NanoJev, a decision model built on Qwen3-0.6B. The writer was `openai/gpt-6.1-sol`. The run used the same task, harness and settings as the d1-omni-600M run in `../2026-11-04-typed-decisions-d1/`, on the same MacBook Pro with MLX.

The writer changed one tensor: `scalar.weight`, the 1,024 weights of NanoJev's scalar head. It computed the change in closed form from the head's inputs, with no gradients. On the 400 Typed Decisions test cases, which the writer never saw, the program's accuracy rose from 38.6% to 55.7%. The paired gain was +17.1 points with a standard error of 1.35 points. The edit improved 243 cases and made 63 worse.

The run took 107 minutes. The writer's session took 85 minutes, with 66 turns, 39 Python runs and 25 forward passes. The final check took about 15 minutes, because NanoJev reads every option on its own text path.

## Setup

The setup matches the d1 run, with these differences:

- The student was `C-Tianyu/NanoJev` at revision `047b927b30882a1138fc504821b82ac145a4b81a`, in float32. It is a Qwen3-0.6B backbone with a decision head. The backbone's final hidden state at each option path's last token goes through a LayerNorm (`norm`) and a scalar head (`scalar`) to give that option's logit. For choice questions, a set-attention correction over the options adds to each logit.
- NanoJev was trained on game decisions (mazes, snake and shooting games), not on business records like these.
- We loaded the model with our own MLX code, `NanoJevBackend`. Before the run, we checked it against NanoJev's own `DecisionModel` on 100 Typed Decisions questions. Our token paths were identical to the reference's, and the largest probability difference was 1.1e-6.
- The code was commit `b05b77f` of this repository.

The command, run from `examples/`:

```bash
OPENAI_API_KEY=... python typed_decisions.py --model nanojev --framework mlx
```

## Baselines

The unedited model on the 400 test cases:

| Request format | Accuracy | KL from gold | Brier | ECE |
|---|---|---|---|---|
| The dataset scorer's protocol | 38.8% | 0.558 | 0.311 | 0.143 |
| The DSPy program | 38.6% | 0.566 | 0.315 | 0.150 |

Both scores are below the 47.0% of the leaderboard's "prior" reference, which ignores the input and predicts each question's most common answer. The two request formats give almost the same score for NanoJev.

## What the writer did

The writer's notes and turns are in `run/`.

1. It ran the original model on all 900 practice cases (36.4%). It found that NanoJev picked nearly the same option for each question whatever the record said: agent trace action was right 1.3% of the time, invoice duplicate 9.3% and customer category 24.4%.
2. It split the practice cases into 600 fitting cases and 300 untouched practice cases. It read the head's inputs on the 600 fitting cases in three read passes.
3. It fit a correction to `scalar.weight` by ridge regression on the first 400 fitting cases. The target was the gold distribution, centred per question, minus the original logits. It predicted the effect of each strength from the read outputs, then confirmed two strengths on 200 fitting cases: a quarter strength gained +6.6 ± 1.2 points and three-quarter strength +16.8 ± 1.9. The measured accuracy matched the prediction at both strengths.
4. It refit the correction on all 600 fitting cases, choosing the target form and the regularization on fitting cases only. On the 300 untouched practice cases, the edit gained +21.3 ± 1.5 points, from 36.2% to 57.5%. On the 300 validation cases it gained +22.4 ± 1.6 points, from 34.8% to 57.2%, with 198 cases improved and 33 worse.
5. It spent the remaining 17 passes on checks of how the edit works. It changed no weights after the validation run and submitted the validated edit. A final validation run measured the same edit again with the same result.

The writer looked at validation results twice, both times for the submitted edit. The test split stayed untouched.

## How the edit works

The writer's checks found two parts to the gain:

- Most of it comes from fixing NanoJev's fixed option preferences. On the 300 untouched practice cases, a version that only adds a fixed offset per question and option reached 51.2%, up from 36.2%.
- The rest comes from reading the record. The full edit reached 57.5% on the same cases, 6.3 points more than the fixed offsets alone.
- Reversing the edit's direction lowered accuracy to 31.5% on a diagnostic set, and increasing its strength raised accuracy until the gain levelled off.

Like the d1 edit, this edit is a closed-form fit of a head readout to practice data. It is close to fitting a linear probe on features the backbone already computes. The new scalar weights are about 100 times larger than the original ones, which were small (norm 0.55). The validation and test results show that it carries over to records the writer never saw.

## Results

On the 400 test cases, through the DSPy program:

| | Accuracy | Standard error |
|---|---|---|
| Original weights | 38.55% | 1.06 |
| Edited weights | 55.65% | 1.16 |
| Paired gain | +17.10 points | 1.35 |

The edit improved 243 cases and made 63 worse. The model is deterministic, so the paired comparison has no sampling noise.

The gain by workflow:

| Workflow | Paired gain on test |
|---|---|
| Invoices | +23.6 ± 2.7 points |
| Agent traces | +23.0 ± 2.5 points |
| Customer service | +11.0 ± 2.5 points |
| Security incidents | +10.8 ± 2.8 points |

The largest gains were on questions where the original model almost always chose the wrong option. Invoice duplicate rose from 11% to 89%. Agent trace action rose from 2% to 39%. Invoice discrepancy severity rose from 16% to 54%. Agent trace review rose from 45% to 79%. Security severity rose from 32% to 66%. Three questions got worse: invoice urgency fell from 53% to 47%, invoice disposition from 41% to 36%, and security disposition from 74% to 71%.

Under the dataset scorer's protocol, the edited model improved on every measure:

| Scorer protocol, 400 test cases | Accuracy | KL from gold | Brier | ECE |
|---|---|---|---|---|
| Original weights | 38.8% | 0.558 | 0.311 | 0.143 |
| Edited weights | 55.2% | 0.324 | 0.172 | 0.042 |

## Files

- `run/report.json` has the budget used, both validation runs, the final check and the patch summary.
- `run/final_answer.md` is the writer's final answer.
- `run/writer_notes.md` holds the notes the writer kept during the session.
- `run/task_prompt.txt` is the task the writer was given.
- `run/transcript.jsonl` lists every Python run and forward pass with its output.
- `run/turns/` holds each turn's prompt and the writer's reply.
- `run/patch/` is the submitted patch. `Patch.load` rebuilds the edited weights from the pinned NanoJev and checks their hashes.
- `run/final_check_scores.jsonl` has one line per test case with both scores and the metric's feedback for both.
- `checks/` holds the two baselines and the edited model under the scorer's protocol.

## How to reproduce

```bash
pip install -e ".[decision,examples]"
cd examples
python typed_decisions.py --model nanojev --framework mlx --baseline protocol
python typed_decisions.py --model nanojev --framework mlx --baseline program
OPENAI_API_KEY=... python typed_decisions.py --model nanojev --framework mlx
```

The writer's answers are sampled, so a new run will make different choices. NanoJev's source code is MIT-licensed; its model card states no license.
