The smaller update matches the first candidate’s practice-test gain: +15.67 percentage points, with a paired SE of 1.30 points. I’ll record that result, inspect the saved candidates, and prepare validation before considering it for the final edit.
```python
# Hypothesis: actual-weight regularization preserves the repair while reducing unnecessary weight movement.
import pathlib, json, numpy as np, torch
work_dir = pathlib.Path("/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/typed-decisions/d1-2b-astra/work")
with (work_dir / "notes.md").open("a") as f:
    f.write("\nPass 17 reserved practice test, n=360: actual-weight-ridge1 candidate original .607778 -> edited .764444; paired gain +.156667 SE .013013; 208 improved, 50 worsened. Same accuracy gain as best_1, with smaller relative weight change 1.8534.\n")
    f.write("\nPlanned pass 18: DECIDE both on hidden validation n=300. Exact current edit: layers.23.feed_forward.w2.weight = W2_original + B A_weightnorm_ridge1 W2_original, alpha1, rank<=9, fit=360. No other tensors changed. best_1 remains preserved.\n")
print("Current edits:", edited())
print("Saved candidates:", sorted(p.name for p in work_dir.iterdir() if p.is_dir() and ("best" in p.name or "candidate" in p.name)))
print("Functions:", sorted(k for k,v in list(globals().items()) if callable(v) and getattr(v, "__module__", "") == "__main__"))
print("Array/tensor variables:", [(k, tuple(v.shape)) for k,v in list(globals().items()) if isinstance(v, (np.ndarray, torch.Tensor)) and not k.startswith("_")])
print("Other analysis containers:", [(k,type(v).__name__,len(v)) for k,v in list(globals().items()) if isinstance(v,(dict,list,tuple)) and not k.startswith("_") and k not in ["NAMES","SHAPES"]])
```