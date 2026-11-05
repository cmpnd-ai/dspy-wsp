"""Our NanoJev loaders against NanoJev's own DecisionModel, on real weights.

Opt-in: set WSP_REFERENCE_TESTS=1. Downloads the pinned NanoJev checkpoint (about
2.4 GB), NanoJev's training script from its source repository, and the Typed
Decisions test split. The reference runs on the CPU in float32.
"""

import importlib.util
import json
import os
import urllib.request

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("WSP_REFERENCE_TESTS") != "1", reason="set WSP_REFERENCE_TESTS=1")

REPO, REVISION = "C-Tianyu/NanoJev", "047b927b30882a1138fc504821b82ac145a4b81a"
SOURCE = "https://raw.githubusercontent.com/TianyuCodings/NanoJev/main/scripts/{}"


def load_reference(tmp_path, name):
    path = tmp_path / name
    urllib.request.urlretrieve(SOURCE.format(name), path)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_parity(tmp_path):
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from dspy_wsp.decision import nanojev
    from dspy_wsp.decision.nanojev_mlx import NanoJevMLX
    from dspy_wsp.decision.nanojev_torch import NanoJevTorch
    from dspy_wsp.decision.schema import parse_question

    folder = snapshot_download(REPO, revision=REVISION, allow_patterns=["best.safetensors", "config.json", "backbone_config/*", "tokenizer/*"])
    rows = pq.read_table(hf_hub_download("LocalLLaMA/typed-decisions", "all/test-00000-of-00001.parquet", repo_type="dataset")).slice(0, 20).to_pylist()
    cases = [(json.loads(r["state"]), json.loads(r["questions"])) for r in rows]

    trainer = load_reference(tmp_path, "train_toy_decisions.py")
    predictor = load_reference(tmp_path, "predict_toy_decisions.py")
    hf_tok = AutoTokenizer.from_pretrained(f"{folder}/tokenizer")
    body_cfg = AutoConfig.from_pretrained(f"{folder}/backbone_config")
    body_cfg.use_cache = False
    body = AutoModel.from_config(body_cfg, attn_implementation="sdpa").float()
    ref = trainer.DecisionModel(body, json.load(open(f"{folder}/config.json"))["set_head"])
    ref.load_state_dict(load_file(f"{folder}/best.safetensors"), strict=True)
    ref.eval()
    native = {"choice": "choice", "score": "score", "noul": "boolean"}
    payload = {"states": [{"id": str(i), "state": s, "questions": {k: {**{x: y for x, y in q.items() if x in ("instructions", "criteria")}, "type": native[q["type"]]} for k, q in qs.items()}} for i, (s, qs) in enumerate(cases)]}
    examples = predictor.prepare_examples(payload, hf_tok, 8192)
    expected = []
    with torch.no_grad():
        for start in range(0, len(examples), 8):
            group = examples[start : start + 8]
            logits, _ = ref(group, hf_tok.pad_token_id)
            expected += [values[: len(ex["candidate_ids"])].float().softmax(-1).tolist() for ex, values in zip(group, logits)]
    del ref, body

    tok = nanojev.Tokenizer(f"{folder}/tokenizer")
    ours_rows = [(*nanojev.encode(tok, s, parse_question(q), 8192), parse_question(q)) for s, qs in cases for q in qs.values()]
    assert [r[0] for r in ours_rows] == [ex["leaf_tokens"] for ex in examples]
    for name, model in (("torch", NanoJevTorch.from_dir(folder, pad_token_id=tok.pad_token_id)), ("mlx", NanoJevMLX.from_dir(folder, pad_token_id=tok.pad_token_id))):
        got = [nanojev.probabilities(z, r[2]) for z, r in zip(model.logits(ours_rows), ours_rows)]
        worst = max(float(np.max(np.abs(np.array(a) - np.array(b)))) for a, b in zip(got, expected))
        print(f"{name}: largest probability difference {worst:.2e} over {len(expected)} questions")
        assert worst < 1e-4, name
