# d1-2B support: design

## Intent

`WeightSpace` edits two decision-model families today: d1-omni-600M (an encoder with a decision head) and NanoJev (a Qwen3 backbone with a scalar head). Both have a separate head, and every edit the writers kept was a closed-form refit of that head.

d1-2B is Liquid AI's current open decision model. It has no separate head. It reads each answer from its language-model output at an answer position, as a softmax over the tokens that code the options. Its output layer is tied to the token embeddings, which the paper's rules protect. A run on d1-2B therefore tests whether a writer can improve a decision model by editing its backbone.

The implementation must be clean, stand-alone and interpretable. A reader should be able to follow the family's code from the request to the probabilities without Liquid's code, transformers' remote code or a custom DSPy adapter. Liquid's code appears only in opt-in parity tests.

## Goals

- Load d1-2B (`LiquidAI/d1-2B`, pinned revision) with our own code, in PyTorch and in MLX.
- Answer Choice, Noul and Score questions with the same probabilities as Liquid's code, within float32 tolerance.
- Support `read` passes, edits, the compatibility check and checkpoint saving, as for the other families.
- Reorganize the decision code into one folder per family, with a registry, so each family reads as one unit.

## Non-goals

- Pictures, video and audio. The family reads text states only.
- d1-2B's other question types: text, number, date, box, boxes and multiple choice. The backend refuses them with a clear error.
- d1-3B (an LFM2-VL backbone) and d1-Omni-2B's audio path. The design should make both straightforward later, and d1-Omni-2B's text path should load with no extra code if its config matches.
- A custom DSPy adapter. DSPy's TypeSafe client builds the request, as for the other families.
- Speed work such as fused projections, varlen attention or torch.compile.

## Structure

```
decision/
  schema.py  server.py  client.py  loading.py     shared
  families/
    __init__.py        the registry
    d1_omni/           d1-omni-600M, moved from d1.py, d1_torch.py, d1_mlx.py
      encoding.py  torch.py  mlx.py
    nanojev/           moved from nanojev.py, nanojev_torch.py, nanojev_mlx.py
      encoding.py  torch.py  mlx.py
    d1_lm/             d1-2B, and later d1-Omni-2B and d1-3B
      encoding.py      request to token ids, attention layout and answer positions; readout; calibration
      lfm3_torch.py    the LFM3 backbone in PyTorch
      lfm3_mlx.py      the LFM3 backbone in MLX
      torch.py         backbone, attention mask and readout: logits(rows), capture(rows, layers)
      mlx.py           the same in MLX
backends/decision.py   DecisionBackend and one subclass per family
```

- Each family's `encoding.py` uses only Python, NumPy and the `tokenizers` library. It turns a request into token ids and records what the network needs. The framework files turn that record into tensors and run the network.
- Module names in `lfm3_torch.py` and `lfm3_mlx.py` equal the checkpoint's tensor names, so edits address checkpoint names directly, as in the other families.
- The registry lists, for each family, how to recognize its config, its backend class and its aliases. `loading.py` reads the registry instead of hardcoding families.
- The existing `d1` family becomes `d1_omni`. `D1Backend` keeps its name, and the alias `"d1"` still opens d1-omni-600M. The new alias `"d1-2b"` opens d1-2B. The new backend is `D1LMBackend`.
- The move is its own commit with no behavior change.

## Backend change: requests as the unit of encoding

`DecisionBackend` encodes each question as its own row today. d1-2B packs up to 16 questions after one copy of the state, and the questions share that sequence. The backend gains `_encode_request(state, questions) -> list[row]`, and each row records which questions it answers. The default calls `_encode` once a question, so d1-omni and NanoJev behave as before. `d1_lm` overrides it.

## The d1-2B input encoding

This follows Liquid's `prompt.py`, `attention.py` and `runner.py` at the pinned revision, for text states and the three question types. The parity test checks that token ids and answer positions are identical.

**Prefix.** The prefix is shared by every question on one state:

```
<|startoftext|><|im_start|>user\n{state block}\nQUESTION:\n
```

- The state block is a string state as written, or any other state as `json.dumps(state, ensure_ascii=False, indent=2)`. It ends with a blank line.
- A zero-width space breaks any special-token string inside the state, so a state that contains `<|im_end|>` does not end the turn.
- No system turn is written.

**Question suffix.** Each question adds this suffix:

```
{question block}<|im_end|>\n<|im_start|>assistant\n<think></think>
```

The answer position is the suffix's last token. The question block depends on the type:

- Choice: `{instructions}\n\nOptions:\n{code} {description or key with _ as spaces}\n...\n\nReply with the option code only.`
  - The codes are the keys themselves when every key is one letter.
  - Otherwise the codes are A, B, C, and so on, for up to 26 options.
  - Otherwise, with `d1_codes` = 2, the codes run A to Z, then AA, AB, and so on.
  - Each code must be a single token. Liquid's fallback pool supplies a replacement when it is not.
- Noul: `{instructions}` and, with criteria, `\nYes: {true}\nNo: {false}`, then `\n\nReply with yes or no only.`
- Score: `{instructions}\n\n0 {level}\n1 {level}\n...\n\nReply with a single digit 0-{n-1} only.`

**Packing.** The questions of a request are taken in order, in groups of `set_size` (16, the trained size).

- One sequence holds the prefix, the first question's suffix, and then `QUESTION:\n` plus the suffix of each further question in the group.
- The encoding records the state length (the prefix's tokens before the first suffix), each token's question number (1 to N on question tokens, -1 on the state) and each question's answer position.

**Attention.** Masks apply to the attention layers only:

- Full-attention layers:
  - a state token reads every real token;
  - every token reads the state;
  - a question token reads the tokens of its own question in both directions.
- Sliding-window layers are causal within 256 tokens.
- Convolution layers are causal and mask padding.
- A group of one question is one segment, and every token reads every token.

## Readout and calibration

- At each answer position, the network applies the final norm and multiplies by the embedding rows of the option tokens. It never computes the full vocabulary.
- Each option scores the larger of two forms, the bare code and the code after a space.
  - Noul scores the best of `yes`, `Yes` and `YES` against the best of `no`, `No` and `NO`.
  - Score uses the digits `0` to `n-1`.
- The probabilities are a softmax of those scores divided by the calibration temperature. For these three types the temperature is `calibration.json`'s default (1.337 at the pinned revision), because its per-type table is empty.
- A Noul's answer maps Liquid's [yes, no] order onto our [false, true] option keys.

## Editing and reads

- Every LFM3 tensor is editable except those that `default_protected` protects:
  - the token embedding, which is also the readout;
  - the final norm.
- The vision tower and the projector are unused and not editable.
- `read` returns, for each question:
  - `option_logits` and `probabilities`;
  - `head_inputs`: the normed hidden state at the answer position, with the option rows it is scored against;
  - `hidden_mean` per requested layer.
- The task prompt's description names the readout, the protected tensors and where closed-form edits can land. The last block's MLP output projection writes straight into the hidden state that the readout scores.

## Tests

- **Tiny fixture.** `tests/decision_fixtures.build_d1_lm` builds a tiny random LFM3 checkpoint. It has conv, sliding and full-attention layers, sandwich norms and a ChatML tokenizer with the special tokens. All unit tests use it and download nothing.
- **Encoding unit tests:**
  - the code schemes and single-token codes;
  - state rendering and special-token breaking;
  - packing and question numbers;
  - answer positions;
  - refusal of unsupported types;
  - the Noul key order.
- **Mask unit tests.** A test checks the mask built from the encoding record against the rules above, cell by cell, for a two-question group.
- **Opt-in parity tests (`WSP_REFERENCE_TESTS=1`).** These run on the real d1-2B:
  - token ids and answer positions equal Liquid's `_sequences` on 100 Typed Decisions requests;
  - our PyTorch probabilities match Liquid's `system_one` in float32 within 1e-4;
  - MLX matches PyTorch within 1e-4.
- **Backend tests.** The same tests as the other families: batching, edits, restoring, reads and saving a checkpoint, then reloading it.
- `dspy-wsp check d1-2b` passes on the real model.

## Left to the builder

- How the MLX and PyTorch ports organize internal helpers, provided module names match the checkpoint.
- How to batch requests of different lengths. Padding to the longest sequence in a batch, with a token budget, is enough.

## Calls made without asking

- The family is named `d1_lm`, because answers are read from the language-model output. The existing family is renamed `d1_omni`.
- The model runs in float32 (about 7.7 GB), as the other families do.
- `set_size` is fixed at 16.
