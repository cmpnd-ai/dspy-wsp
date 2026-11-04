# A one-hour MATH-500 run with Qwen3.5-0.8B on eight GPUs

This folder documents a proof of concept. We ran `WeightSpace` for one hour on the paper's Experiment 8 task to show that the harness works end to end on GPUs. The writer was `openai/gpt-6-luna`. The student was Qwen3.5-0.8B with thinking on. The writer had 8 forward passes, where the paper used 50.

The writer found no edit it trusted, so it submitted the original weights. The final check on 250 MATH-500 problems compared the original weights with themselves. The original weights scored 8.4% and the submitted weights scored 8.8%, a paired difference of +0.4 points with a standard error of 1.6 points. That difference measures how much two runs of the same weights disagree.

The run used the paper's settings for the student and the prompt. It was not meant to reproduce the paper's result, and it does not. The paper's writer needed 25 passes to raise MATH-500 from 7.9% to 14.1%, and 50 passes to reach 48.1%.

## Setup

- The hardware was one Lambda Cloud instance with 8 NVIDIA A100 SXM4 GPUs of 40 GB each, NVIDIA driver 580.105.08 and Ubuntu 24.04.5.
- The software was vLLM 0.33.0, PyTorch 2.14.0 with CUDA 13.0, transformers 5.20.2, DSPy 3.4.1, math-verify 0.9.0, bubblewrap 0.9.0 and Python 3.12.3.
- The code was commit `2e6a572` of this repository.
- The student was `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`. It ran with thinking on, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5 and at most 32,768 new tokens. These are the paper's settings.
- The prompt was the paper's prompt. It is one user message with the problem followed by "Please reason step by step, and put your final answer within \boxed{}." The metric grades the last `\boxed{}` answer anywhere in the reply with math-verify, as the paper's evaluation does.
- The backend was `VLLMBackend`. It ran one copy of the model on each GPU, which vLLM calls data parallelism, and gave vLLM 80% of each GPU's memory. Teacher-forced reads ran on a Hugging Face copy of the model on GPU 0.
- The writer's code ran under bubblewrap. It had no network, and it could write only to its work folder.

The command, run from `examples/` on the instance:

```bash
python math500_qwen_vllm.py --dp 8 --gpu-mem 0.8 --paper-prompt \
    --passes 8 --pass-size 256 --val-size 250 --val-samples 1 --run-dir runs/hour
```

A pass could hold up to 256 answers, or 128 per model for a paired "both" pass. The writer practiced on the paper's 7,275 MATH train problems. The final check used 250 MATH-500 problems with one sample each. The writer never saw them.

## Checks before the run

We ran `examples/smoke_vllm.py` on the same instance before the run. Its output for eight GPUs is in `checks/smoke_vllm_8gpu.json`.

- Sandbox. Code inside bubblewrap could write its work folder. It could not write the home folder, and it could not open a network connection.
- Weight swaps. We added random noise to two MLP tensors in block 12. vLLM stores one of them, `gate_proj`, together with `up_proj` in a single tensor. For each of the eight model copies, we compared vLLM's log-probabilities of four texts with the Hugging Face model's log-probabilities, averaged per token and measured in nats. The worst copy differed from Hugging Face by 0.017 nats per token on the original weights and 0.027 on the edited weights. The edit moved every copy by at least 0.418 nats per token, and it moved the Hugging Face model by 0.409. After the edit was removed, every copy matched its original log-probs exactly. A swap took 0.8 seconds.
- Speed. The eight GPUs generated 27,000 to 28,000 tokens per second. A pass of 256 answers that run to the token limit takes about 5 minutes.
- Capacity. Each GPU had room to store attention keys and values (the KV cache) for 2,025,521 tokens. That is enough for 49 answers of full length at once.

We also measured the unedited model on 256 MATH-500 problems with two prompts. The numbers are in `checks/baselines.json`.

| Prompt | Score | Answers cut at the token limit |
|---|---|---|
| The paper's prompt (used for this run) | 9.0% ± 1.8 | 91.8% |
| DSPy's `ChatAdapter` prompt | 2.7% ± 1.0 | 97.3% |
| The paper, unedited model (500 problems, 4 samples each) | 7.9% | 93.0% |

With the paper's prompt, our unedited score matches the paper's within one standard error. With DSPy's default prompt, the model loops more often and scores lower. `ChatAdapter` adds a system message with field markers, and it only reads the text after `</think>`, so an answer cut off while thinking scores 0 even when its thinking already holds the right boxed answer.

## The run

The run started at 23:03:13 UTC on 2026-11-04 and ended at 23:23:48 UTC, 20.6 minutes in all. The writer's session took about 10 minutes, with 49 turns and 24 Python runs. After turn 40, the harness restarted the conversation once and gave the writer its notes. The final check took about 10 minutes.

The table lists the writer's passes. "Block 23, columns 9 and 46" means the weights of `mlp.down_proj` in block 23 that carry MLP channels 9 and 46 back into the model's hidden state.

| Pass | Request | Edit tested | Result |
|---|---|---|---|
| 1 | Original weights, 8 problems, 2,048 tokens | None | 0 of 8 correct. All 8 cut at the limit. |
| 2 | Original weights, 4 problems, 512 tokens, with pooled activations | None | 0 of 4. All cut. |
| 3 | Both models, 4 problems, 512 tokens | Block 23, columns 9 and 46, at half strength | 0 of 4 for both. All cut. |
| 4 | Both models, 4 problems, 2,048 tokens | Block 23, columns 9 and 46, set to zero | 0 of 4 for both. All cut. |
| 5 | Teacher-forced read of 2 of its own answers (3,363 tokens), MLP inputs of all 24 blocks | None | The writer found that MLP channel 382 of block 3 was more active near repeated text. |
| 6 | Both models, 4 problems, 2,048 tokens | Block 3, column 382, set to zero | Original 0 of 4. Edited 1 of 4. |
| 7 | Both models, 24 new problems, 2,048 tokens | Block 3, column 382, set to zero | Original 0 of 24. Edited 1 of 24. All cut. |
| 8 | Both models, 8 new problems, 32,768 tokens | Block 3, column 382, set to zero | Original 2 of 8. Edited 1 of 8. Cut answers rose from 6 to 7. |

After pass 8, the writer removed the edit. Its final answer reads, "The edit's evidence was inconsistent: it gained 1/24 on the short-limit paired test but all calls still cut off; on the full-limit test it lost by 1/8, with more cutoffs (7/8 versus 6/8). Given the small samples and negative full-budget result, I found no justified edit to retain."

## Results

| | Score | Answers cut at the token limit |
|---|---|---|
| Original weights | 8.4% (21 of 250) | 92.4% |
| Submitted weights (the same as the original) | 8.8% (22 of 250) | 91.6% |
| Paired difference | +0.4 points, standard error 1.6 | 8 problems improved, 7 got worse |

The two sets of weights are identical, and each problem used the same random seed for both. Fifteen problems still changed outcome. vLLM put different answers together in each batch for the two runs. With different batches, the 16-bit arithmetic rounds differently in the last bits. A few sampled tokens then differ, and the rest of the answer differs after them. For this check, a paired gain needs to exceed about 3 points before it means anything.

## What we learned

- The writer spent its passes on small tests. Seven of its eight passes used 4 to 24 problems, and six of them stopped answers at 512 or 2,048 tokens. At those limits almost every answer is cut off, so the score is 0 for nearly every problem and the tests could not show a gain. With 8 passes, the writer did not have the budget to repeat a test at full size.
- The prompt changes the baseline. The paper's prompt and grading give 9.0%. DSPy's `ChatAdapter` gives 2.7%. Use `--paper-prompt` to compare results with the paper.
- The paired comparison has a noise floor. Identical weights differed by 8 improved and 7 worse problems out of 250. Any claim about an edit needs a gain well above that.
- A full run at the paper's settings would take about 10 to 15 hours on this hardware. A generation pass of 512 answers at full length takes about 10 minutes, and a final check of 500 problems with 4 samples each takes about 75 minutes.

## Changes made while preparing this run

We made these changes after testing on the GPUs. All of them are in this repository's history.


- `VLLMBackend` and its worker extension, which reload the full checkpoint with edited tensors in place and then clear vLLM's prefix cache.
- `--dp` for data parallelism, and a smoke test that checks every model copy after a swap.
- File requests for the forward-pass tool. In an earlier 3-pass test, the writer could not paste its own 32,000-token answers into a teacher-forced request, so it sent placeholder strings. The writer can now save a request as JSON in its work folder and send `{"file": "<path>"}`. It used this in pass 5 of this run.
- `PaperPromptAdapter` and `--paper-prompt`.
- The setup script installs FlashInfer's prebuilt kernels, because vLLM otherwise compiles them at startup and needs `nvcc` and `ninja`.

## Files

- `run/report.json` has the budget used, the final check and the patch summary.
- `run/final_answer.md` is the writer's final answer.
- `run/writer_notes.md` holds the notes the writer kept during the session.
- `run/task_prompt.txt` is the task the writer was given.
- `run/transcript.jsonl` lists every Python run and forward pass with its output.
- `run/turns/` holds each turn's prompt and the writer's reply.
- `run/final_check_scores.jsonl` has one line per final-check problem with both scores, output lengths and whether each answer was cut. The full answers (39 MB) are not in the repository.
- `run/patch_manifest.json` describes the submitted patch, which changes no weights.
- `checks/smoke_vllm_8gpu.json` holds the sandbox, weight swap and speed checks.
- `checks/baselines.json` holds the two baseline measurements.

## How to reproduce

On a fresh Lambda Cloud instance with 8 GPUs, from a clone of this repository at commit `2e6a572` or later:

```bash
bash examples/lambda_setup.sh
cd examples
../.venv/bin/python smoke_vllm.py --gpu-mem 0.8 --dp 8 --n 256
OPENAI_API_KEY=... ../.venv/bin/python math500_qwen_vllm.py --dp 8 --gpu-mem 0.8 --paper-prompt \
    --passes 8 --pass-size 256 --val-size 250 --val-samples 1 --run-dir runs/hour
```

Keep the API key file outside the writer's view. The sandbox mounts the whole disk read-only but replaces `/tmp` with an empty folder, so a key file in `/tmp` is hidden from the writer's code. The writer's answers are sampled, so a new run will make different choices.
