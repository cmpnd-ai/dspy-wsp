I’ll verify that the saved head inputs reproduce the scorer’s logits. If they do, I can test a precise intervention in the final readout without changing or simulating the encoder.

```python
# Hypothesis: the last linear readout is a bottleneck; verify exact scorer reconstruction before editing it.
print({k:v for k,v in meta.items() if k!="rows"})
torch.set_num_threads(4)
sw={n:load_original(n) for n in NAMES if "head.scorer" in n}
def features(h):
    h=torch.as_tensor(h)
    h=torch.nn.functional.layer_norm(h,(1024,),sw["head.scorer.0.weight"],sw["head.scorer.0.bias"],1e-5)
    return torch.nn.functional.gelu(torch.nn.functional.linear(h,sw["head.scorer.1.weight"],sw["head.scorer.1.bias"]))
h=np.load(work/"fwd_2/head_inputs.npy")
z=features(h)
raw=torch.nn.functional.linear(z,sw["head.scorer.3.weight"],sw["head.scorer.3.bias"]).squeeze(-1).numpy()
actual=np.load(work/"fwd_2/option_logits.npy")
mask=np.load(work/"fwd_2/option_mask.npy").astype(bool)
print("Reconstruction error",np.max(abs(raw[mask]-actual[mask])))
print("Raw/logit examples", raw[:2],actual[:2])
```
