I’ll first refit only the simpler two-neuron repair, keeping its hard-margin target, ridge regularization, and half-strength setting unchanged. The opposite GELU neurons implement a linear correction exactly, so this remains a localized readout edit rather than a new decision architecture.



I’ll refit only the simpler two-neuron repair first, keeping its target, regularization, and half-strength setting unchanged. The opposite GELU neurons implement a linear correction exactly, so this remains a localized readout edit.


```python
# Hypothesis: the validated two-neuron direction will be more stable when estimated once from all 900 practice records.
repair_data13=repair_read_data(13)
FULL={}
for k,source in [("features",repair_features),("logits",repair_logits),("mask",repair_mask),
                 ("gold",repair_gold),("target",repair_target)]:
    FULL[k]=torch.cat([source,repair_data12[k],repair_data13[k]],dim=0)
for k in ["meta","ids","qkey"]:
    FULL[k]=D[k]+repair_data12[k]+repair_data13[k]
assert len(set(FULL["ids"]))==900 and len(FULL["ids"])==4500
FULL["nt"]=FULL["mask"].sum(-1).float()
FULL["fc"]=center_options(FULL["features"][:,:,1024:],FULL["mask"])
FULL["lc"]=center_options(torch.nan_to_num(FULL["logits"],neginf=0),FULL["mask"])
FULL["hard"]=center_options(F.one_hot(FULL["target"],5).float()*2.5,FULL["mask"])
full_x=(FULL["fc"]/FULL["nt"][:,None,None].sqrt())[FULL["mask"]].double()
full_y=((FULL["hard"]-FULL["lc"])/FULL["nt"][:,None].sqrt())[FULL["mask"]].double()
full_gram=full_x.T@full_x
full_rhs=full_x.T@full_y
full_ridge=1e-4*float(full_gram.trace()/full_gram.shape[0])
full_direction=torch.linalg.solve(full_gram+torch.eye(3072,dtype=torch.float64)*full_ridge,full_rhs).float()
full_gain=float(repair_base1_tensors[2].flatten()[repair_pair[0]])
full_linear=[repair_W0.clone(),repair_b0.clone(),repair_wout.clone()]
full_row=torch.zeros(4096)
full_row[1024:]=.5*full_direction/(repair_gate*full_gain)
for ii,sgn in zip(repair_pair,[1,-1]):
    full_linear[0][ii]=sgn*full_row
    full_linear[1][ii]=0
    full_linear[2][ii]=sgn*full_gain
full_linear=tuple(x.bfloat16().float() for x in full_linear)
def full_simulate(tensors):
    rw,rb,ro=tensors
    ro=ro.flatten()
    changed=((rw!=repair_W0).any(1)|(rb!=repair_b0)|(ro!=repair_wout)).nonzero().flatten()
    anew=F.gelu(F.linear(FULL["features"],rw[changed],rb[changed]))
    aold=F.gelu(F.linear(FULL["features"],repair_W0[changed],repair_b0[changed]))
    eff=repair_gate*((anew*ro[changed]).sum(-1)-(aold*repair_wout[changed]).sum(-1))
    return (FULL["logits"]+eff).masked_fill(~FULL["mask"],-torch.inf)
def full_measure(logits,reference=None):
    reference=FULL["logits"] if reference is None else reference
    correct=(logits.masked_fill(~FULL["mask"],-torch.inf).argmax(-1)==FULL["target"]).float()
    diff=correct-(reference.masked_fill(~FULL["mask"],-torch.inf).argmax(-1)==FULL["target"]).float()
    per=collections.defaultdict(list)
    for ii,exid in enumerate(FULL["ids"]): per[exid].append(float(diff[ii]))
    vals=np.array([np.mean(v) for v in per.values()])
    return dict(acc=float(correct.mean()),gain=float(vals.mean()),se=float(vals.std(ddof=1)/np.sqrt(len(vals))),n=len(vals))
full_linear_logits=full_simulate(full_linear)
print("Full refit direction norm:",float(full_direction.norm()),"ridge",full_ridge)
print("Cosine with 500-fit direction:",float(F.cosine_similarity(full_direction,repair_b1["delta"],dim=0)))
print("In-sample practice only:",full_measure(full_linear_logits))
for n,a in zip(["residual_scorer.0.weight","residual_scorer.0.bias","residual_scorer.3.weight"],full_linear):
    save_edit(n,a.reshape(SHAPES[n]))
torch.save({"tensors":full_linear,"direction":full_direction,"ridge":full_ridge,"ids":sorted(set(FULL["ids"]))},WORK/"full_linear_candidate.pt")
del full_x,full_y,full_gram,full_rhs
with open(WORK/"notes.md","a") as f:
    f.write("\n## Full-practice fixed-design two-neuron refit\n900 records, original option-centered features. Target = centered 2.5*onehot(gold argmax) minus original centered logits. Ridge = 1e-4*trace(Gram)/3072, strength 0.5; neurons 791/1015, opposite rows and outgoing weights. No hyperparameter search. In-sample practice: "+str(full_measure(full_linear_logits))+". This is NOT an independent test.\n")
    f.write("\n## Planned pass 14\nExact edits: full_linear_candidate.pt, only two neurons in residual_scorer.0.weight/.bias and residual_scorer.3.weight. DECIDE both on all 300 hidden validation items. best_2 remains backed up.\n")
```
