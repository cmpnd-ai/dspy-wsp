from pathlib import Path

import dspy
import numpy as np
import pytest
import torch
from dspy.lm15 import Message, Response, TextPart, Usage, response_to_events
from safetensors.torch import save_file

from dspy_wsp.backends.base import GenerativeBackend, LMCall

TENSORS = {
    "model.embed_tokens.weight": (10, 4),
    "model.layers.0.mlp.down_proj.weight": (4, 8),
    "model.layers.1.mlp.down_proj.weight": (4, 8),
    "model.norm.weight": (4,),
}


@pytest.fixture
def weights_dir(tmp_path) -> Path:
    g = torch.Generator().manual_seed(0)
    d = tmp_path / "model"
    d.mkdir()
    tensors = {n: torch.randn(s, generator=g).to(torch.bfloat16) for n, s in TENSORS.items()}
    save_file(tensors, str(d / "model.safetensors"), metadata={"format": "pt"})
    (d / "config.json").write_text("{}")
    return d


class FakeBackend(GenerativeBackend):
    """Answers "4" when any edit is applied and "5" otherwise, so edits visibly matter."""

    def __init__(self, weights_dir):
        super().__init__()
        self._dir = Path(weights_dir)
        self._lm = None
        self.applied = []

    @property
    def model_id(self):
        return "fake"

    @property
    def weights_dir(self):
        return self._dir

    def _apply(self, edits):
        self.applied.append(sorted(edits))

    def lm(self):
        if self._lm is None:
            self._lm = dspy.LM("wsp/fake", engine=FakeEngine(self), cache=False, max_tokens=64)
        return self._lm

    def seed(self, seed):
        pass

    def generate(self, prompt, **kw):
        raise NotImplementedError("FakeEngine answers without the backend")

    def render_chat(self, messages, *, add_generation_prompt=False, **kw):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def num_tokens(self, text):
        return len(text.split())

    def dims(self):
        return {"layers": 2, "hidden": 4, "intermediate": 8, "heads": 2}

    def teacher_forced(self, texts, *, token_logprobs=False, topk=0, hidden_states=(), mlp_act=(), attn_layers=()):
        n, T = len(texts), max(len(t.split()) for t in texts)
        arrays = {"token_ids": np.zeros((n, T), np.int32), "lengths": np.array([len(t.split()) for t in texts], np.int32)}
        if token_logprobs:
            arrays["token_logprobs"] = np.full((n, T), -1.0 if self.current_edits else -2.0, np.float32)
        if hidden_states:
            arrays["hidden_states"] = np.zeros((n, len(hidden_states), T, 4), np.float16)
        return arrays, {"texts": n}

    def pooled_activations(self, pairs, *, layers, mlp_layers=(), max_answer_tokens=8192):
        out = {}
        if layers:
            out["hidden_last_prompt"] = np.zeros((len(pairs), len(layers), 4), np.float32)
        if mlp_layers:
            out["mlp_last_prompt"] = np.zeros((len(pairs), len(mlp_layers), 8), np.float32)
        return out


class FakeEngine:
    def __init__(self, backend):
        self.backend = backend

    def complete(self, request):
        ans = "4" if self.backend.current_edits else "5"
        text = f"[[ ## answer ## ]]\n{ans}\n\n[[ ## completed ## ]]"
        prompt = "\n".join(m.text or "" for m in request.messages)
        self.backend._record(LMCall(prompt=prompt, completion=text, finish_reason="stop", input_tokens=10, output_tokens=5))
        return Response(id=None, model="fake", message=Message.assistant([TextPart(text)]), finish_reason="stop",
                        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15))

    def stream(self, request):
        return response_to_events(self.complete(request))

    def close(self):
        pass


class ScriptedEngine:
    """A writer that replies from a script and records what it was sent."""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.received = []

    def complete(self, request):
        self.received.append([(m.role, m.text or "") for m in request.messages])
        text = next(self.replies, "Done. I have nothing more to add.")
        return Response(id=None, model="scripted", message=Message.assistant([TextPart(text)]), finish_reason="stop",
                        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2))

    def stream(self, request):
        return response_to_events(self.complete(request))

    def close(self):
        pass


def scripted_writer(replies):
    engine = ScriptedEngine(replies)
    return dspy.LM("wsp/scripted-writer", engine=engine, cache=False, max_tokens=1000), engine


@pytest.fixture
def backend(weights_dir):
    return FakeBackend(weights_dir)


def exact_answer(example, pred, trace=None):
    return float(str(pred.answer).strip() == str(example.answer))


@pytest.fixture
def trainset():
    return [dspy.Example(question=f"What is {i} + {4 - i}?", answer="4").with_inputs("question") for i in range(5)]


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    """A tiny random Llama with its own tokenizer, saved in Hugging Face format (no downloads)."""
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("tiny")
    corpus = ["[[ ## question ## ]] What is 2 + 2? [[ ## answer ## ]] 4 [[ ## completed ## ]]", "Your input fields are: question. Respond with answer."] * 20
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "<s>", "</s>", "<pad>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>")
    fast.chat_template = "{% for m in messages %}<s>{{ m['role'] }}\n{{ m['content'] }}</s>\n{% endfor %}{% if add_generation_prompt %}<s>assistant\n{% endif %}"
    fast.save_pretrained(d)
    cfg = LlamaConfig(vocab_size=len(fast), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=4096, tie_word_embeddings=False,
                      bos_token_id=fast.bos_token_id, eos_token_id=fast.eos_token_id, pad_token_id=fast.pad_token_id)
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(d)
    return d
