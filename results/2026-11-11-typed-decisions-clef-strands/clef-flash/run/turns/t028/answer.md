I’ll check whether the three proposed targeted fields can be isolated reliably using their existing question–option representations. The validated edit remains unchanged during this diagnostic.

```python
# Hypothesis: question-specific state evidence can improve agent urgency and customer handling without perturbing unrelated decisions.
target_qkeys = [
    ("agent_trace_observability","urgency"),
    ("customer_service","action"),
    ("customer_service","needs_human")
]
option_classes=[]
for qkey in qkeys:
    i=D["qkey"].index(qkey)
    option_classes += [(qkey,o) for o in D["meta"][i]["options"]]
class_index={c:i for i,c in enumerate(option_classes)}
class_ids=torch.full(D["mask"].shape,-1,dtype=torch.long)
for i,(qkey,m) in enumerate(zip(D["qkey"],D["meta"])):
    for j,o in enumerate(m["options"]):
        class_ids[i,j]=class_index[qkey,o]
gate_features=D["features"][:,:,:2048].double()
fit_flat_mask=D["mask"] & D["isfit"][:,None]
gate_mu=gate_features[fit_flat_mask].mean(0)
centroids=torch.stack([gate_features[(class_ids==c)&fit_flat_mask].mean(0)-gate_mu for c in range(len(option_classes))])
cg=centroids@centroids.T
gate_lambda=cg.diag().mean().item()*1e-5
gate_coef=centroids.T@torch.linalg.solve(cg+gate_lambda*torch.eye(len(option_classes),dtype=torch.double),
                                       torch.eye(len(option_classes),dtype=torch.double))
gate_scores=(gate_features-gate_mu)@gate_coef
print("Number of option identities:",len(option_classes))
print("Prototype identity accuracy fit/tune:",
      (gate_scores.argmax(-1)==class_ids)[fit_flat_mask].float().mean().item(),
      (gate_scores.argmax(-1)==class_ids)[D["mask"]&D["istune"][:,None]].float().mean().item())
gates={}
for c,(qkey,o) in enumerate(option_classes):
    if qkey not in target_qkeys:
        continue
    pos=(class_ids==c)&fit_flat_mask
    neg=(class_ids!=c)&fit_flat_mask
    lo=gate_scores[:,:,c][pos].min().item()
    hi=gate_scores[:,:,c][neg].max().item()
    threshold=(lo+hi)/2
    margin=(lo-hi)/2
    v=gate_coef[:,c]/margin
    bias=(-gate_mu@gate_coef[:,c]-threshold)/margin
    gates[c]=(v.float(),float(bias))
    tune_pos=(class_ids==c)&D["mask"]&D["istune"][:,None]
    tune_neg=(class_ids!=c)&D["mask"]&D["istune"][:,None]
    ts=(gate_scores[:,:,c]-threshold)/margin
    print(qkey[0][:7],qkey[1],o,"fit margin",round(margin,3),"tune min+/max-",
          round(ts[tune_pos].min().item(),2),round(ts[tune_neg].max().item(),2))
targetmask=torch.tensor([q in target_qkeys for q in D["qkey"]])
ideal_targeted=edited_pair_logits(.5).clone()
ideal_targeted[targetmask]=conditional_L[targetmask]
print("Ideal targeted replacement:",summary_logits(ideal_targeted,D["istune"]))
note("## Targeted extension plan\nPreselected three fields from tune mechanism comparison: agent urgency; customer action and needs_human. Replace shared correction only there with soft-target field read (ridge 0.01), fitted on 125 records/workflow. Prototype question-option gates use fit features, not gold, and are checked on tune. Other fields retain best_1.")
```

This diagnostic changes no weights; the validation-confirmed edit remains the fallback.