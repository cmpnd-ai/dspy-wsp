The current scorer reacts to a changed price, but barely distinguishes a 1% difference from a 40% difference; it also fails to raise discrepancy severity for delivery exceptions. I’ll test whether directions isolated by these controlled changes can improve the readout. Any coefficients will be solved once from the original fit labels, not from invented gold answers for the probes.
```python
# Hypothesis: a low-dimensional subspace isolated by paired invoice cues separates useful mismatch evidence from nuisance context.
print("Target reconstruction error:",float((target_mode - 5*D["temp"][:,None]*centered(.75*D["gold"]+.25*F.one_hot(D["gold"].argmax(-1),5).float())).abs().max()))
invoice_qnames=["discrepancy_severity","disposition","matches_order"]
cue_directions={}
cue_candidate_adds={rel:torch.zeros_like(D["logits"]) for rel in [0.01,0.1,1.0]}
cue_solutions={rel:{} for rel in cue_candidate_adds}
cue_residual=target_mode-centered(cond_logits)
for qname in invoice_qnames:
    qi=qkeys.index(("invoice_processing",qname))
    fitq=(qid==qi)&D["fit"]
    allq=(qid==qi)
    r15=[j for j,r in enumerate(m15["rows"]) if r["question"]==qname]
    nopt=int(D["n"][fitq][0])
    for oi in range(nopt):
        cfmat=x15[r15,oi].reshape(8,11,1024)
        # Each row is a matched-pair cue difference averaged across source invoices.
        contrasts=(cfmat[:,1:]-cfmat[:,:1]).mean(0)
        _,scue,Vcue=torch.linalg.svd(contrasts,full_matrices=False)
        basis=Vcue.T
        cue_directions[(qi,oi)]=(basis,scue)
        xf=D["x"][fitq,oi]
        xm=xf.mean(0)
        features=(xf-xm)@basis
        yr=cue_residual[fitq,oi]
        ym=yr.mean()
        gram=features.T@features/len(features)
        rhs=features.T@(yr-ym)/len(features)
        for rel in cue_candidate_adds:
            coeff=torch.linalg.solve(gram+rel*gram.trace()/len(gram)*torch.eye(len(gram)),rhs)
            dd=basis@coeff
            bb=float(ym-xm@dd)
            cue_solutions[rel][(qi,oi)]=(dd,bb)
            cue_candidate_adds[rel][allq,oi]=D["x"][allq,oi]@dd+bb
print("Cue-restricted residual on dev (screening only):")
for rel,aa in cue_candidate_adds.items():
    print(rel, {s:evaluate_logits(cond_logits+s*aa) for s in [.5,1.]})
print("Baseline current",evaluate_logits(cond_logits))
print("Gate pair values:",[(i,float(cond_params[2][0,cond_units[i]])) for i in range(4)])
print("Contrast spectra severity option0:",cue_directions[(qkeys.index(("invoice_processing","discrepancy_severity")),0)][1].tolist())
```
