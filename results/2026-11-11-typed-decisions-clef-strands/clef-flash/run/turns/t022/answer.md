Question-specific field reads alone do not beat the validated edit. I’ll test a narrower hypothesis: different questions need different reads of the *field–option interaction features*. I’ll also inspect the scorer’s singular spectrum to quantify the compression rather than infer it only from performance.

```python
# Hypothesis: the shared scorer collapses question-specific field–option evidence into a nearly common scoring axis.
sv = torch.linalg.svdvals(W0)
print("Scorer input-weight top singular values:",sv[:8].tolist())
print("Energy captured by top 1/4/16:",[(sv[:k].square().sum()/sv.square().sum()).item() for k in [1,4,16]])
print("Number of baseline neurons below option-contribution RMS thresholds:",
      {t:int((nrms<t).sum()) for t in [1e-7,1e-5,1e-4,1e-3]})
qraw={}
for qkey in qkeys:
    ix=torch.tensor([k==qkey for k in D["qkey"]])
    tr=ix & D["isfit"]
    m=D["mask"][tr]
    n=m.sum(-1).float()
    x=(D["FC"][tr]/n.sqrt()[:,None,None])[m].double()
    gram=x@x.T
    ev,U=torch.linalg.eigh(gram)
    scale=gram.diag().mean().item()
    qraw[qkey]=dict(ix=ix,tr=tr,x=x,mask=m,n=n,ev=ev,U=U,scale=scale)
qr_candidates={}
print("Question-conditioned interaction readout: target ridge strength fit tune gain SE")
for tname in ["soft","hard"]:
    residual=target_sets[tname]-center_options(D["logits"],D["mask"])
    for ridge in [1e-5,1e-4,1e-3,.01,.1]:
        correction=torch.zeros_like(D["logits"])
        directions={}
        for qkey,qd in qraw.items():
            yy=(residual[qd["tr"]]/qd["n"].sqrt()[:,None])[qd["mask"]].double()
            dd=qd["x"].T@(qd["U"]@((qd["U"].T@yy)/(qd["ev"]+ridge*qd["scale"])))
            correction[qd["ix"]]=(D["FC"][qd["ix"]]@dd.float())
            directions[qkey]=dd.float()
        qr_candidates[tname,ridge]={"correction":correction,"directions":directions}
        for strength in [.5,1.]:
            L=D["logits"]+strength*correction
            ff,tt=summary_logits(L,D["isfit"]),summary_logits(L,D["istune"])
            print(tname,ridge,strength,*[round(v,4) for v in [ff["accuracy"],tt["accuracy"],tt["gain"],tt["se"]]])
```
