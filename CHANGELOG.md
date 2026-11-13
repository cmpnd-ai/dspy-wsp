# Changelog

## 0.2.0

- `ClefBackend` loads Cloudflare's Clef decision models in MLX, with the alias `"clef-flash"`. The backbone stays fixed and the writer edits the joint schema head.
- `StrandsBackend` loads Strands decider models in MLX, with the alias `"strands-2b"`. It folds the checkpoint's LoRA adapter into the base model's weights and saves the result in `~/.cache/dspy-wsp`, so the writer can edit the backbone and the head.
- The package requires Python 3.11 or later. `mlx-lm` needs it on Apple silicon.
- The README shows how to save an edited model and load it in a new file.
- `results/2026-11-11-typed-decisions-clef-strands/` documents the Clef-Flash and Strands Decider 2B runs, and the results chart shows all five models.
- A decision model's family can be recognized without a `config.json` at the checkpoint's root.
- The tensor index covers safetensors files outside a checkpoint's shard map.
- `examples/typed_decisions.py --no-checkpoint` keeps the patch and skips the full copy of the edited model.
- `--baseline protocol` works again on DSPy 3.5.

## 0.1.2

- The benchmark scripts in `examples/` use `gpt-6-astra` as the writer by default. Pass `--writer sol` for `gpt-6.1-sol`. Run folder names are unchanged.
- `dspy_wsp.__version__` reads the installed package's version. It reported `0.1.0` in the 0.1.1 release.
- The sandbox is an allowlist. The writer's kernel reads only the system's libraries, the Python install, this package and the model's files. It read the whole disk before. `sandbox_readonly=[...]` adds folders.
- `sandbox="auto"` works on macOS, through `sandbox-exec`. The examples no longer pass `sandbox="none"` on a Mac.
- The host refuses symlinks and hard links for the writer's `notes.md` and edit files.
- `WeightSpace(...)` checks the sandbox when it is built. A missing sandbox now raises before the run loads a model.
- `WeightBackend` states its whole contract. `GenerativeBackend` is the new base class for text-generating backends: it holds `generate`, `render_chat`, `teacher_forced` and `pooled_activations`. A custom backend that implements those methods must now subclass `GenerativeBackend`. `DecisionBackend` no longer carries them.
- Backends are context managers: `with load_decision_backend("d1") as backend:` closes the backend at the end.
- The package requires DSPy 3.5 (`dspy>=3.5,<3.6`). Decision models use DSPy's built-in TypeSafe provider, so `typesafe-sdk` is no longer a dependency.
- `gepa` is a declared dependency.
- The README covers the Ubuntu 24.04 sandbox setup.

## 0.1.1

- `simulate_head=False` keeps the paper's rule for decision models: the writer runs the network only through the forward-pass tool.
- The README reports the d1-omni-600M run under the paper's rule.

## 0.1.0

- First release: the `WeightSpace` optimizer, decision backends for d1-omni, NanoJev and d1-2B, the `HFBackend`, `MLXBackend` and `VLLMBackend` LLM backends, GEPA population search, and the patch format with its CLI.
