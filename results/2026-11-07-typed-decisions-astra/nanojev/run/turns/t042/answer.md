The completed cache should let me refit the same correction on all 900 practice records. I’ll verify the row alignment and saved recipe first; misaligning an option with its label would invalidate the closed-form solve.
```python
# Hypothesis: the new read files can extend the existing fit without changing its target, regularization, or option alignment.
import numpy as np, torch
print("Analysis callables:")
for k,v in list(globals().items()):
    if inspect.isfunction(v) and k not in ["load","load_original","save_edit","drop_edit","edited"]:
        print(k, inspect.signature(v), getattr(v,"__module__", None))
print("Other relevant containers:")
for k,v in list(globals().items()):
    if not k.startswith("_") and isinstance(v,(dict,list)) and k not in ["NAMES","SHAPES","head0"]:
        print(k, type(v).__name__, len(v))
print("Row examples:", rows[0], rows[-1])
print("Split keys:", [(k, len(v), v[:2]) for k,v in splits.items()])
print("READ16 metadata:", json.loads((work/"fwd_16"/"meta.json").read_text())["rows"][:2] if "rows" in json.loads((work/"fwd_16"/"meta.json").read_text()) else json.loads((work/"fwd_16"/"meta.json").read_text()))
print("Old meta keys:", meta2.keys())
print("Norm and ridge values:", {k:v for k,v in list(globals().items()) if isinstance(v,(int,float,str)) and any(s in k.lower() for s in ["ridge","lam","scale","strength","alpha"])})
print("Feature/target samples:", X.shape, G[:2].round(3), mask[:2], counts[:2], RW[:2], Tmix[:2].round(3), Y[:2].round(3), L0[:2].round(3))
print("Scalar shapes/norms:", np.linalg.norm(w0), np.linalg.norm(d_mix600), head0.keys())
```
