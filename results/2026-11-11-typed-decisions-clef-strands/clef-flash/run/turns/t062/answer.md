The full-practice two-neuron refit improved validation, but less than the saved 24-neuron candidate. I won’t keep it on that basis. I’ll complete the preplanned fixed-design refit of the three targeted fields, then compare the complete candidate with the saved best.


```python
# Hypothesis: the three validated field-specific readouts may benefit from more data even though the shared-direction refit did not.
print("Target gate models:")
for kk,vv in list(gate_models.items())[:2]:
    print(kk, {a:(tuple(b.shape) if isinstance(b,torch.Tensor) else str(b)[:100]) for a,b in vv.items()} if isinstance(vv,dict) else [(type(a).__name__,tuple(a.shape) if isinstance(a,torch.Tensor) else str(a)[:100]) for a in vv])
print("Target param tuple:",[(type(x).__name__,tuple(x.shape) if isinstance(x,torch.Tensor) else x) for x in conditional_candidates[("soft",.01)]["param"][target_qkeys[0]]])
print("Target candidate metadata:",repair_b2["neurons"],repair_b2["qkeys"])
print("Function constants:",targeted_tensors.__code__.co_names,targeted_tensors.__code__.co_consts)
print("Gate globals:",[(k,tuple(v.shape) if isinstance(v,torch.Tensor) else type(v).__name__) for k,v in globals().copy().items() if k in targeted_tensors.__code__.co_names])
import dis, io, contextlib
stream=io.StringIO()
with contextlib.redirect_stdout(stream): dis.dis(targeted_tensors)
print(stream.getvalue()[:8500])
with open(WORK/"notes.md","a") as f:
    f.write("\n## Pass 14 full-data two-neuron refit validation\nOriginal 0.702000; edited 0.755333; gain +0.053333, paired SE 0.007920, n=300, 91 improved / 23 worsened. Below best_2's 0.770667, so not selected. Will complete the previously planned fixed-design refit of the three targeted fields, then judge full candidate once; no validation-based hyperparameter search.\n")
```
