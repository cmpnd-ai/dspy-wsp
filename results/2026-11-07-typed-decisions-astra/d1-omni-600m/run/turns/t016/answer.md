The likely repair site is now specific: the scorer’s input retains useful option distinctions, while its MLP readout underuses them. I’ll compare a final set of closed-form targets aligned with the accuracy metric, then freeze a candidate for intervention tests. I’ll also inspect records for controlled cue changes.
```python
# Hypothesis: emphasizing the gold mode in the same closed-form residual readout improves ambiguous top-option ranking, not merely calibration.
onehot=F.one_hot(D["gold"].argmax(1),5).float()*D["mask"]
bone=XX.T@((centered(onehot)*D["temp"][:,None]/D["n"].sqrt()[:,None])[fitmask].double())
bovec=Vx.T@bone
mode_solutions={}
for rel in [0.01,0.03,0.1,0.3]:
    lam=float(ex.sum()/1024)*rel
    for hardmix in [0.25,0.5,1.0]:
        d=(Vx@((5*((1-hardmix)*bpvec+hardmix*bovec)-blvec)/(ex+lam))).float()
        mode_solutions[(rel,hardmix)]=d
        l=D["logits"]+D["x"]@d
        print("mode target",rel,hardmix,"norm",round(float(d.norm()),2),
              "fit",round(evaluate_logits(l,"fit")["acc"],4),
              "dev",{k:round(v,5) for k,v in evaluate_logits(l).items()})
for wf in sorted(split_ids):
    ix=split_ids[wf]["fit"][0]
    print("TEMPLATE",wf,ix,json.dumps(by_id[ix]["inputs"]["state"])[:2300])
    if wf=="security_incidents":
        print("Severity question:",by_id[ix]["labels"]["questions"]["severity"])
best_mode_key=max(mode_solutions,key=lambda k:evaluate_logits(D["logits"]+D["x"]@mode_solutions[k])["acc"])
print("Best mode candidate",best_mode_key)
note("Low-contribution scorer units identified from fit option-contrast RMS: neuron 33 (4.4e-8) and 746 (5.9e-8). Their near-zero original output weights permit an exact +/- GELU linear residual channel with negligible removal effect. No edits yet. Final closed-form target comparison includes soft/mode mixtures; test and validation remain unused.")
```
This comparison still leaves the weights unchanged; the next stage will test the chosen direction causally.
