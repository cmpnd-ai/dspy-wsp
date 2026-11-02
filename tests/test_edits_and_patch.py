import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from dspy_wsp.edits import DEFAULT_PROTECTED, EditError, EditStore, check_edit, save_array
from dspy_wsp.patch import Patch, PatchError
from dspy_wsp.tensors import TensorReader

L0 = "model.layers.0.mlp.down_proj.weight"
L1 = "model.layers.1.mlp.down_proj.weight"


def test_check_edit_rules(weights_dir):
    r = TensorReader(weights_dir)
    o = r.get_f32(L0)
    with pytest.raises(EditError, match="out of scope"):
        check_edit("model.embed_tokens.weight", r.get_f32("model.embed_tokens.weight") + 1, r.get_f32("model.embed_tokens.weight"))
    with pytest.raises(EditError, match="out of scope"):
        check_edit("model.norm.weight", r.get_f32("model.norm.weight") + 1, r.get_f32("model.norm.weight"))
    with pytest.raises(EditError, match="multiplies the whole tensor"):
        check_edit(L0, o * 1.5, o)
    with pytest.raises(EditError, match="equals the original"):
        check_edit(L0, o.copy(), o)
    with pytest.raises(EditError, match="shape"):
        check_edit(L0, o[:2], o)
    bad = o.copy()
    bad[0, 0] = np.nan
    with pytest.raises(EditError, match="NaN"):
        check_edit(L0, bad, o)
    e = o.copy()
    e[:, 3] = 0
    s = check_edit(L0, e, o)
    assert s["changed_entries"] == 4 and 0 < s["relative_change"] < 1


def test_layer_norms_are_not_protected():
    from dspy_wsp.edits import is_protected

    assert is_protected("model.norm.weight", DEFAULT_PROTECTED)
    assert is_protected("model.language_model.norm.weight", DEFAULT_PROTECTED)
    assert not is_protected("model.layers.3.input_layernorm.weight", DEFAULT_PROTECTED)
    assert not is_protected("model.layers.3.self_attn.q_norm.weight", DEFAULT_PROTECTED)


def test_edit_store_catches_edits_written_around_save_edit(weights_dir, tmp_path):
    r = TensorReader(weights_dir)
    store = EditStore(tmp_path / "edited", r)
    save_array(store.dir, "model.norm.weight", r.get_f32("model.norm.weight") + 1)
    with pytest.raises(EditError):
        store.load_all()


def _edits(r):
    sparse = r.get_f32(L0)
    sparse[:, 2] -= 0.5  # 4 of 32 entries
    dense = r.get_f32(L1) + 0.25 * np.arange(32, dtype=np.float32).reshape(4, 8)
    return {L0: sparse, L1: dense}


def test_patch_roundtrip_and_hashes(weights_dir, tmp_path):
    r = TensorReader(weights_dir)
    edits = _edits(r)
    patch = Patch.from_edits(r, edits, meta={"note": "test"})
    forms = {t.name: t.form for t in patch.tensors}
    assert forms == {L0: "sparse", L1: "dense"}
    patch.save(tmp_path / "patch")
    loaded = Patch.load(tmp_path / "patch")
    rebuilt = loaded.edited_tensors(r)
    for name, arr in edits.items():
        expected = torch.from_numpy(arr).to(torch.bfloat16)
        assert torch.equal(rebuilt[name], expected)
    assert loaded.meta["note"] == "test"
    assert "2 tensors" in loaded.summary()


def test_patch_refuses_a_different_base(weights_dir):
    r = TensorReader(weights_dir)
    patch = Patch.from_edits(r, _edits(r))
    wrong = r.get(L0).clone()
    wrong[0, 0] += 1
    with pytest.raises(PatchError, match="differs"):
        patch.tensors[0].rebuild(wrong)


def test_write_checkpoint(weights_dir, tmp_path):
    r = TensorReader(weights_dir)
    edits = _edits(r)
    patch = Patch.from_edits(r, edits)
    out = patch.write_checkpoint(weights_dir, tmp_path / "edited_model")
    got = load_file(str(out / "model.safetensors"))
    assert torch.equal(got[L0], torch.from_numpy(edits[L0]).to(torch.bfloat16))
    assert torch.equal(got["model.norm.weight"], r.get("model.norm.weight"))
    assert (out / "config.json").exists() and (out / "wsp_patch.json").exists()


def test_rounding_away_is_not_a_change(weights_dir):
    r = TensorReader(weights_dir)
    tiny = r.get_f32(L0)
    tiny[0, 0] += 1e-6  # vanishes in bfloat16
    assert len(Patch.from_edits(r, {L0: tiny})) == 0
