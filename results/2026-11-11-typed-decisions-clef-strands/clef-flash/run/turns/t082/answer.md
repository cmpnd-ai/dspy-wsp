The saved best edit remains unchanged. Pass 22 collected cue probes from practice seeds excluded from its fit and tuning sets. I’ll compare original and repaired logits on those probes, then verify the predictions with an edited-model read.


I’ll also check that the submitted tensors still match the saved candidate before running the comparison.


This inspection is read-only; it won’t change or fit any weights.


```python
# Hypothesis: best_2 remains intact, and cached helpers can isolate its response on independent-seed probes.
import inspect, json, numpy as np, torch
print("Relevant cached functions:")
for _name, _obj in list(globals().items()):
    if inspect.isfunction(_obj) and any(s in _name.lower() for s in ("probe", "simulat", "logit", "note", "fingerprint", "measure")):
        try:
            print(_name, inspect.signature(_obj))
        except Exception:
            pass

for _name in ("best1", "best2", "W0", "b0", "gate", "held_probe_meta", "held_probe_requests", "held_seed_ids"):
    _obj = globals().get(_name)
    if isinstance(_obj, torch.Tensor):
        print(_name, "tensor", tuple(_obj.shape))
    elif isinstance(_obj, dict):
        print(_name, "dict keys", list(_obj)[:8])
    elif isinstance(_obj, list):
        print(_name, "list length", len(_obj), "first", str(_obj[:1])[:600])
    else:
        print(_name, type(_obj).__name__, str(_obj)[:150])

_meta22 = json.loads((_handoff_work / "fwd_22/meta.json").read_text())
print("Pass22 metadata type:", type(_meta22).__name__)
if isinstance(_meta22, dict):
    print("Metadata keys:", list(_meta22))
    for _key, _value in _meta22.items():
        print(_key, str(_value)[:900])
print("Edited tensors:", edited())
```
