I’ll check exactly what the read tool saved and verify that the scorer can reproduce the original logits. That lets me test a small readout edit locally, while leaving the encoder and transformer head fixed.
```python
# Hypothesis: the existing nonlinear scorer features contain useful cue distinctions that its final linear readout underweights.
meta2=json.load(open(WORK/"fwd_2/meta.json"))
print("META:",json.dumps(meta2,indent=2)[:6500])
arr2={p.stem:np.load(p,mmap_mode="r") for p in (WORK/"fwd_2").glob("*.npy")}
print("ARRAYS:",{k:(a.shape,str(a.dtype)) for k,a in arr2.items()})
print("HEAD WEIGHTS:",[(n,SHAPES[n]) for n in NAMES if n.startswith("head.")])
print("Head code exact:",encsrc[encsrc.index("class DecisionHead"):encsrc.index("def _rotate_half")])
W0=load_original("head.scorer.3.weight").clone()
scorer_params={n:load_original("head.scorer."+n).clone() for n in ["0.weight","0.bias","1.weight","1.bias","3.weight","3.bias"]}
import torch.nn.functional as F
def scorer_features(head_inputs):
    x=torch.as_tensor(np.array(head_inputs),dtype=torch.float32)
    x=F.layer_norm(x,(1024,),scorer_params["0.weight"],scorer_params["0.bias"],1e-5)
    a=F.linear(x,scorer_params["1.weight"],scorer_params["1.bias"])
    return F.gelu(a),x,a
z2,x2,a2=scorer_features(arr2["head_inputs"])
lp2=F.linear(z2,scorer_params["3.weight"],scorer_params["3.bias"]).squeeze(-1).numpy()
mask2=np.asarray(arr2["option_mask"],dtype=bool)
print("Reconstruction max absolute error:",np.max(np.abs(lp2[mask2]-arr2["option_logits"][mask2])))
print("Original final scorer weight norm:",float(W0.norm()),"range",float(W0.min()),float(W0.max()))
json.dump({"type":"read","model":"original","from":{"pass":1,"ids":read_groups[1]}},
          open(WORK/"read_fit2.json","w"))
note("Diagnosis: fit/dev accuracies (672 examples): agent risk 39.9%, outcome 42.9%; invoice discrepancy severity 40.5%, urgency 42.3%; security severity 32.7%. Repeated confusions: invoice urgency gold elevated->routine 77/168, discrepancy gold material/severe->none 89/168; agent harmful->partial 33/168. High-confidence examples show explicit cues (overdue, prohibited freight, constraint violations) ignored or underused.\n"
     "Pass 2 READ: 200 balanced fit examples, 1000 questions. Testing final scorer readout reconstruction, no backbone simulation.\n"
     "Next pass: READ original second 200 balanced fit examples (1000 questions), exact IDs in read_fit2.json. Exact edits: none.")
```
