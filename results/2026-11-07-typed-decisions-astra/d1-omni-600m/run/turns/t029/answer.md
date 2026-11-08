The first repair leaves substantial invoice errors. I’ll test whether a single shared readout is forced to compromise between workflows: the same hidden direction may need a different interpretation for invoices than for traces or security alerts. Any conditional repair must identify the workflow from existing activations, not from an added input.
```python
# Hypothesis: cross-workflow interference limits the shared residual channel, and frozen activations support reliably gated workflow-specific corrections.
cfactual=torch.tensor(np.nan_to_num(np.load(WORK/"fwd_10/option_logits.npy"),nan=0))
def cf_center(a):
    return (a-(a*cfmask).sum(1,keepdim=True)/cfmask.sum(1)[:,None])*cfmask
print("Counterfactual actual vs prediction max contrast error:",float((cf_center(cfactual)-cf_center(cflogits)).abs().max()))
cfvactual=cf_values(cfactual)
for family,low,high,q in cfcontrasts:
    ids=sorted({t["id"] for t in cf_tags if t["family"]==family})
    b=np.array([cfvb[family,i,high,q]-cfvb[family,i,low,q] for i in ids])
    e=np.array([cfvactual[family,i,high,q]-cfvactual[family,i,low,q] for i in ids])
    print("Actual",family,q,round(b.mean(),4),"->",round(e.mean(),4))
workflows=sorted(split_ids)
wfidx=torch.tensor([workflows.index(r["workflow"]) for r in D["rows"]])
base_linear=D["logits"]-pair_removed+D["x"]@candidate_dx
target_mode=centered(5*D["temp"][:,None]*(0.75*D["gold"]+0.25*onehot))
local_directions={}
for wfno,wf in enumerate(workflows):
    fm=fitmask&(wfidx==wfno)[:,None]
    xw=(Xc/D["n"].sqrt()[:,None,None])[fm].double()
    yw=((target_mode-centered(base_linear))/D["n"].sqrt()[:,None])[fm].double()
    cw=xw.T@xw
    ew,vw=torch.linalg.eigh(cw)
    bw=vw.T@(xw.T@yw)
    local_directions[wf]={}
    for rel in [0.003,0.01,0.03,0.1,0.3,1]:
        dw=(vw@(bw/(ew+float(ew.sum()/1024)*rel))).float()
        local_directions[wf][rel]=dw
        l=base_linear.clone(); l[wfidx==wfno]+=D["x"][wfidx==wfno]@dw
        ev=evaluate_logits(l)
        print("Workflow residual",wf[:8],rel,"norm",round(float(dw.norm()),2),
              "dev acc",round(ev["acc"],4),"gain over current",round(ev["acc"]-evaluate_logits(base_linear)["acc"],4))
# Fit workflow identity gates from labels that are already in the question/state; no gold answers used here.
XA=D["x"]-mu_x
A=(XA/D["n"].sqrt()[:,None,None])[fitmask].double()
CA=A.T@A
ea,Va=torch.linalg.eigh(CA)
gate_labels=(2*F.one_hot(wfidx,len(workflows)).float()-1)
gate_mean=gate_labels[D["fit"]].mean(0)
Ygate=((gate_labels-gate_mean)[:,None,:].expand(-1,5,-1)/D["n"].sqrt()[:,None,None])[fitmask].double()
gateW=(Va@((Va.T@(A.T@Ygate))/(ea[:,None]+float(ea.sum()/1024)*0.01))).float()
gates=XA@gateW+gate_mean
for w,wf in enumerate(workflows):
    valid=D["dev"][:,None]&D["mask"]
    pos=valid&(wfidx==w)[:,None]; neg=valid&(wfidx!=w)[:,None]
    print("Gate",wf[:8],"positive min/q05",float(gates[:,:,w][pos].min()),float(torch.quantile(gates[:,:,w][pos],.05)),
          "negative max/q95",float(gates[:,:,w][neg].max()),float(torch.quantile(gates[:,:,w][neg],.95)),
          "error",int((gates[:,:,w][pos]<0).sum())+int((gates[:,:,w][neg]>0).sum()))
note("Pass 10 edited counterfactual replay matches local predictions (max option-contrast error recorded above). Actual mean violation->harmful probability contrast .0945 -> .4163, risk expected-level contrast .4106 -> 1.0435, overdue urgency .0429 -> .3792. Discount and freight remain unrepaired. Investigating workflow-specific residual directions using fit data only; current submitted weights still validated best_1.")
```
The validated edit remains unchanged while this analysis checks whether workflow-specific readouts offer a defensible next step.