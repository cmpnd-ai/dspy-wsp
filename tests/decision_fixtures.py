"""Tiny random decision-model checkpoints, built locally so tests download nothing."""

from __future__ import annotations

import json
from pathlib import Path

import torch

D1_SPECIALS = ["<|pad|>", "<|startoftext|>", "<|im_end|>", "<|mask|>"] + [f"<|reserved_{i}|>" for i in range(7, 12)]
CORPUS = [
    "The agent deleted personal data. Halt the agent now. Queue this trace for a human to review.",
    "Invoice overdue: total 1250 USD, vendor Acme. Is the total above 1000? level 0 level 1 level 2",
    "true false yes no choice score noul state question option decision billing technical urgent",
    '{"task": "refund", "status": "open", "steps": 7, "errors": 1}',
] * 30
D1_LM_CORPUS = CORPUS + ["yes Yes YES no No NO QUESTION Options Reply with the option code only"] * 40


def build_tokenizer(folder: Path, specials: list[str], bos: str, pad: str, eos: str | None = None, corpus: list[str] = CORPUS) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus, trainer)
    folder.mkdir(parents=True, exist_ok=True)
    tok.save(str(folder / "tokenizer.json"))
    cfg = {"bos_token": bos, "pad_token": pad}
    if eos:
        cfg["eos_token"] = eos
    (folder / "tokenizer_config.json").write_text(json.dumps(cfg))


def d1_config(vocab: int) -> dict:
    return {
        "model_type": "d1_omni",
        "dtype": "float32",
        "max_length": 512,
        "head_layers": 1,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "temperatures": {"choice:2": 1.7, "choice:3-5": 1.4, "choice:6-10": 1.2, "choice:11+": 1.4, "noul:2": 1.6,
                         "score:3-5": 1.7, "score:6-10": 1.0, "choice": 1.0, "score": 1.0, "noul": 1.0},
        "text_config": {
            "vocab_size": vocab, "hidden_size": 64, "intermediate_size": 192, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "layer_types": ["conv", "full_attention", "conv", "full_attention"],
            "norm_eps": 1e-5, "conv_L_cache": 3, "block_ffn_dim_multiplier": 1.0, "block_multiple_of": 32,
            "rope_theta": 1_000_000.0,
        },
    }


def build_d1(folder: Path) -> Path:
    """A tiny d1 checkpoint folder: config, tokenizer and random weights, plus unused media tensors."""
    from safetensors.torch import save_file

    from dspy_wsp.decision.families.d1_omni.torch import D1Torch

    build_tokenizer(folder, D1_SPECIALS, bos="<|startoftext|>", pad="<|pad|>")
    from tokenizers import Tokenizer

    vocab = Tokenizer.from_file(str(folder / "tokenizer.json")).get_vocab_size()
    cfg = d1_config(vocab)
    (folder / "config.json").write_text(json.dumps(cfg))
    torch.manual_seed(0)
    model = D1Torch(cfg)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                p.normal_(0, 0.2)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state["vision.patch_embed.weight"] = torch.zeros(8, 3)
    state["audio.proj.weight"] = torch.zeros(4, 4)
    save_file(state, str(folder / "model.safetensors"))
    return folder


NANOJEV_SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def nanojev_backbone_config(vocab: int) -> dict:
    return {
        "model_type": "qwen3", "vocab_size": vocab, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-6, "rope_theta": 1_000_000.0,
        "max_position_embeddings": 4096, "tie_word_embeddings": False, "hidden_act": "silu", "attention_bias": False,
    }


def build_nanojev(folder: Path) -> Path:
    """A tiny NanoJev checkpoint folder: run config, backbone config, tokenizer and random weights."""
    from safetensors.torch import save_file
    from tokenizers import Tokenizer

    from dspy_wsp.decision.families.nanojev.torch import NanoJevTorch

    build_tokenizer(folder / "tokenizer", NANOJEV_SPECIALS, bos="<|im_start|>", pad="<|endoftext|>", eos="<|im_end|>")
    vocab = Tokenizer.from_file(str(folder / "tokenizer" / "tokenizer.json")).get_vocab_size()
    (folder / "backbone_config").mkdir(parents=True, exist_ok=True)
    backbone = nanojev_backbone_config(vocab)
    (folder / "backbone_config" / "config.json").write_text(json.dumps(backbone))
    (folder / "config.json").write_text(json.dumps({"set_head": "attention", "max_length": 512, "model": "tiny-qwen3"}))
    torch.manual_seed(0)
    model = NanoJevTorch(backbone, "attention")
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                p.normal_(0, 0.2)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(folder / "best.safetensors"))
    return folder


D1_LM_SPECIALS = ["<|pad|>", "<|startoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]


def build_d1_lm_tokenizer(folder: Path) -> Path:
    """A ChatML byte-level tokenizer in which yes, Yes, no and No are single tokens."""
    build_tokenizer(folder, D1_LM_SPECIALS, bos="<|startoftext|>", pad="<|pad|>", eos="<|im_end|>", corpus=D1_LM_CORPUS)
    return folder


def d1_lm_config(vocab: int) -> dict:
    """A tiny d1-2B config: every LFM3 layer kind, sandwich norms, partial RoPE on full attention."""
    return {
        "model_type": "lfm3_vl", "architectures": ["D1Model"], "d1_attention": "set", "d1_set_size": 16, "d1_codes": 2,
        "tie_word_embeddings": True, "dtype": "float32",
        "text_config": {
            "model_type": "lfm3", "vocab_size": vocab, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 6, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "layer_types": ["conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention"],
            "conv_L_cache": 7, "conv_bias": False, "sliding_window": 8, "use_sandwich_norm": True, "norm_eps": 1e-5,
            "rope_parameters": {"full_attention": {"rope_theta": 1_000_000.0, "rope_type": "default", "partial_rotary_factor": 0.5},
                                "sliding_attention": {"rope_theta": 10_000.0, "rope_type": "default"}},
            "tie_word_embeddings": True,
        },
    }


def build_d1_lm(folder: Path) -> Path:
    """A tiny d1-2B checkpoint folder: config, calibration, tokenizer and random weights, plus an unused vision tensor."""
    from safetensors.torch import save_file
    from tokenizers import Tokenizer

    from dspy_wsp.decision.families.d1_lm.torch import D1LMTorch

    build_d1_lm_tokenizer(folder)
    vocab = Tokenizer.from_file(str(folder / "tokenizer.json")).get_vocab_size()
    cfg = d1_lm_config(vocab)
    (folder / "config.json").write_text(json.dumps(cfg))
    (folder / "calibration.json").write_text(json.dumps({"default": 1.337, "temperatures": {}}))
    torch.manual_seed(0)
    model = D1LMTorch(cfg)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                p.normal_(0, 0.2)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state["model.vision_tower.dummy.weight"] = torch.zeros(4, 4)
    save_file(state, str(folder / "model.safetensors"))
    return folder
