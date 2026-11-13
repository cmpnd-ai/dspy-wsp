The read files contain the final field/option features, which lets me test scorer-only interventions without rerunning the backbone. I’ll recover the existing diagnostic code and compare field-level results before proposing another edit.



```python
# Hypothesis: remaining errors can be localized by comparing shared versus question-specific readouts of the same frozen features.
import builtins
print("Variables:", [(k,type(v).__name__, tuple(v.shape) if isinstance(v,(torch.Tensor,np.ndarray)) else (len(v) if isinstance(v,(dict,list,set)) else "")) for k,v in builtins.list(globals().items()) if not k.startswith("_")][:250])
for name in ["pair_tensors","edited_pair_logits","targeted_tensors","simulate_scoring_tensors"]:
    fn = globals().get(name)
    if fn is not None:
        try:
            print("\n",name,"\n",inspect.getsource(fn))
        except Exception as e:
            print(name,repr(e))
print("Candidate keys:", list(conditional_candidates)[:30], list(qr_candidates)[:20])
print("Target fields:",target_qkeys)
print("Meta row:",meta2["rows"][0])
print("Example call:",str(r10[0]["calls"])[:2200])
```
