The gated extension is too sensitive to bfloat16 rounding and leakage, so I won’t keep it. I’ll return to the simpler validated mechanism and test a more conservative target: correct deficient gold-option margins while leaving already-large correct margins alone. This may avoid unnecessary changes to confident decisions.

```python
# Hypothesis: a minimum-margin correction preserves good decisions better than forcing every option toward a fixed target logit.
px,pm=[],[]
fitidx=torch.where(D["isfit"])[0].tolist()
pair_rows=[]
for i in fitidx:
    n=int(D["mask"][i].sum())
    t=int(D["target"][i])
    for j in range(n):
        if j!=t:
            px.append((D["features"][i,t,1024:]-D["features"][i,j,1024:])/math.sqrt(n-1))
            pm.append((D["logits"][i,t]-D["logits"][i,j]).item())
            pair_rows.append((i,j,n-1))
PX=torch.stack(px).double()
PG=PX.T@PX
pev,pU=torch.linalg.eigh(PG)
punit=PG.diag().mean().item()
margin_candidates={}
print("Margin ridge strength fit tune gain SE")
for margin in [.25,.75,1.5]:
    py=torch.tensor([max(0.,margin-m)/math.sqrt(nm1) for m,(_,_,nm1) in zip(pm,pair_rows)],dtype=torch.double)
    prhs=PX.T@py
    for ratio in [1e-5,1e-4,1e-3]:
        dd=(pU@((pU.T@prhs)/(pev+ratio*punit))).float()
        margin_candidates[margin,ratio]=dd
        for strength in [.5,1.]:
            wf,bf,wl=pair_tensors(strength,delta=dd)
            new=F.gelu(F.linear(D["features"],wf[pair],bf[pair]))@wl[pair]
            old=D["activations"][:,:,pair]@wout[pair]
            L=D["logits"]+gate*(new-old)
            f,t=summary_logits(L,D["isfit"]),summary_logits(L,D["istune"])
            print(margin,ratio,strength,*[round(v,4) for v in [f["accuracy"],t["accuracy"],t["gain"],t["se"]]])
note("Gated extension not saved: although ideal targeted replacement reached 78.8% tune, concrete gates caused rounding error or leakage; best concrete gain insufficient to justify complexity. Testing a one-shot minimum-margin least-squares correction in the same two-neuron pathway. No iterative fitting; all targets computed once from original logits.")
```
