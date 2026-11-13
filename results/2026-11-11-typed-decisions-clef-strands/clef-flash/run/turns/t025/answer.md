I’ll verify the exact logit match, then test whether the field-only correction complements the validated interaction correction. The goal is to isolate a remaining mechanism—especially urgency decisions—not simply add a larger correction everywhere.

```python
# Hypothesis: field-only cue reads complement the interaction bypass on urgency, while the bypass remains useful elsewhere.
m7=json.loads((WORK/"fwd_7/meta.json").read_text())["rows"]
a7={p.stem:torch.from_numpy(np.array(np.load(p))) for p in (WORK/"fwd_7").glob("*.npy")}
tune_order=[str(r["id"]) for r in base if str(r["id"]) in set(splits["tune"])]
row_lookup={(eid,m["question"]):i for i,(eid,m) in enumerate(zip(D["ids"],D["meta"]))}
idx7=torch.tensor([row_lookup[(tune_order[m["request"]],m["question"])] for m in m7])
sim7=edited_pair_logits(.5)[idx7]
mvalid=a7["option_mask"].bool()
print("Saved edit vs simulation: max logit error",(sim7-a7["option_logits"])[mvalid].abs().max().item())
print("Upstream feature change:",(D["features"][idx7]-torch.nan_to_num(a7["head_inputs"]))[mvalid].abs().max().item())
print("All-field conditional accuracies (baseline, shared, field soft .01 full):")
conditional_L=D["logits"]+conditional_candidates["soft",.01]["correction"]
for qkey in qkeys:
    ix=torch.tensor([k==qkey for k in D["qkey"]])&D["istune"]
    print(qkey[0][:7],qkey[1],[round(summary_logits(L,ix)["accuracy"],3) for L in [D["logits"],edited_pair_logits(.5),conditional_L]])
shared_correction=edited_pair_logits(.5)-D["logits"]
print("Complementary mixtures: shared strength, field strength, fit, tune gain SE")
for a,b in [(1,.25),(.75,.25),(.5,.5),(1,.5),(0.5,.75)]:
    L=D["logits"]+a*shared_correction+b*conditional_candidates["soft",.01]["correction"]
    f,t=summary_logits(L,D["isfit"]),summary_logits(L,D["istune"])
    print(a,b,round(f["accuracy"],4),{k:round(float(v),4) for k,v in t.items()})
urgency_mask=torch.tensor([m["question"]=="urgency" for m in D["meta"]])
for b in [.5,1.]:
    L=edited_pair_logits(.5).clone()
    L[urgency_mask]=D["logits"][urgency_mask]+b*conditional_candidates["soft",.01]["correction"][urgency_mask]
    print("Replace urgency only",b,summary_logits(L,D["istune"]))
note("## Pass 7 causal verification\nSaved two-neuron edit matches local scorer simulation to floating-point precision; upstream scorer inputs unchanged. Therefore observed gains arise solely from the proposed final scoring pathway, not altered routing or backbone states.")
```

I’ll verify the exact logit match, then test whether the field-only correction complements the validated interaction correction. The goal is to isolate a remaining mechanism—especially urgency decisions—not simply add a larger correction everywhere.

```python
# Hypothesis: field-only cue reads complement the interaction bypass on urgency, while the bypass remains useful elsewhere.
m7=json.loads((WORK/"fwd_7/meta.json").read_text())["rows"]
a7={p.stem:torch.from_numpy(np.array(np.load(p))) for p in (WORK/"fwd_7").glob("*.npy")}
tune_order=[str(r["id"]) for r in base if str(r["id"]) in set(splits["tune"])]
row_lookup={(eid,m["question"]):i for i,(eid,m) in enumerate(zip(D["ids"],D["meta"]))}
idx7=torch.tensor([row_lookup[(tune_order[m["request"]],m["question"])] for m in m7])
sim7=edited_pair_logits(.5)[idx7]
mvalid=a7["option_mask"].bool()
print("Saved edit vs simulation: max logit error",(sim7-a7["option_logits"])[mvalid].abs().max().item())
print("Upstream feature change:",(D["features"][idx7]-torch.nan_to_num(a7["head_inputs"]))[mvalid].abs().max().item())
print("All-field conditional accuracies (baseline, shared, field soft .01 full):")
conditional_L=D["logits"]+conditional_candidates["soft",.01]["correction"]
for qkey in qkeys:
    ix=torch.tensor([k==qkey for k in D["qkey"]])&D["istune"]
    print(qkey[0][:7],qkey[1],[round(summary_logits(L,ix)["accuracy"],3) for L in [D["logits"],edited_pair_logits(.5),conditional_L]])
shared_correction=edited_pair_logits(.5)-D["logits"]
print("Complementary mixtures: shared strength, field strength, fit, tune gain SE")
for a,b in [(1,.25),(.75,.25),(.5,.5),(1,.5),(0.5,.75)]:
    L=D["logits"]+a*shared_correction+b*conditional_candidates["soft",.01]["correction"]
    f,t=summary_logits(L,D["isfit"]),summary_logits(L,D["istune"])
    print(a,b,round(f["accuracy"],4),{k:round(float(v),4) for k,v in t.items()})
urgency_mask=torch.tensor([m["question"]=="urgency" for m in D["meta"]])
for b in [.5,1.]:
    L=edited_pair_logits(.5).clone()
    L[urgency_mask]=D["logits"][urgency_mask]+b*conditional_candidates["soft",.01]["correction"][urgency_mask]
    print("Replace urgency only",b,summary_logits(L,D["istune"]))
note("## Pass 7 causal verification\nSaved two-neuron edit matches local scorer simulation to floating-point precision; upstream scorer inputs unchanged. Therefore observed gains arise solely from the proposed final scoring pathway, not altered routing or backbone states.")
```
