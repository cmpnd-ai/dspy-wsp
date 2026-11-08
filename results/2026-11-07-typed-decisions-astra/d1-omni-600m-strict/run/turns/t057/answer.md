The counterfactuals reveal a clear readout mismatch: adding production/secret privileges slightly *reduces* the high-versus-medium logit margin, while crown-jewel assets usually still receive medium severity. Both cues produce reproducible changes in the head inputs. I’ll isolate those directions while removing components shared with unrelated questions.
```python
# Hypothesis: cue-by-severity directions can repair impact judgments while nuisance projection protects other workflows.
# Nuisance means use discovery A only; no fitting to confirmation or validation labels.
nonsec_means = []
nonsec_contrasts = []
for wf,qid in sorted({(r["wf"],r["q"]) for r in rowinfo2 if r["wf"]!="security_incidents"}):
    inds = [r["i"] for r in rowinfo2 if r["wf"]==wf and r["q"]==qid]
    nopt = len(rowinfo2[inds[0]]["opts"])
    for j in range(nopt):
        nonsec_means.append(U2[inds,j].mean(0))
    for i in inds:
        centered = U2[i,:nopt]-U2[i,:nopt].mean(0)
        nonsec_contrasts.extend(centered)
_,_,sec_vh = np.linalg.svd(np.stack(nonsec_contrasts),full_matrices=False)
sec_nuisance_common = list(nonsec_means) + list(sec_vh[:32])
for crit in ["low","medium","high","crown_jewel"]:
    for priv in [False,True]:
        for qid in ["credential_compromise","disposition","true_positive","urgency"]:
            ix=[sec_cf_rows[(eid,crit,priv,qid)] for eid in security_seed_ids]
            nopt=len(meta14["rows"][ix[0]]["options"])
            for j in range(nopt):
                sec_nuisance_common.append(U14[ix,j].mean(0))
def null_project_direction(raw, nuisance):
    mat=np.stack(nuisance).astype(np.float64)
    _,sv,vh=np.linalg.svd(mat,full_matrices=False)
    basis=vh[sv>max(sv[0]*1e-6,1e-9)]
    resid=raw.astype(np.float64)-basis.T@(basis@raw)
    return (resid/np.linalg.norm(resid)).astype(np.float32), float(np.linalg.norm(resid)),len(basis)
crown_ns=list(sec_nuisance_common)
for crit in ["low","medium","high"]:
    for priv in [False,True]:
        ix=[sec_cf_rows[(eid,crit,priv,"severity")] for eid in security_seed_ids]
        crown_ns.extend(U14[ix].mean(0))
d_crown,crown_resnorm,crown_rank=null_project_direction(crit_diffs.mean(0),crown_ns)
prod_ns=list(sec_nuisance_common)
for crit in ["low","medium"]:
    ix=[sec_cf_rows[(eid,crit,False,"severity")] for eid in security_seed_ids]
    prod_ns.extend(U14[ix].mean(0))
# Protect the critical-vs-high distinction from the production-access readout.
for priv in [False,True]:
    ix=[sec_cf_rows[(eid,"crown_jewel",priv,"severity")] for eid in security_seed_ids]
    prod_ns.append((U14[ix,4]-U14[ix,3]).mean(0))
d_prod,prod_resnorm,prod_rank=null_project_direction(priv_diffs.mean(0),prod_ns)
d_high,high_resnorm,high_rank=null_project_direction(high_diffs.mean(0),prod_ns)
print("Protected residual norms/ranks crown,prod,high:",(crown_resnorm,crown_rank),(prod_resnorm,prod_rank),(high_resnorm,high_rank))
print("Protected cos prod/high:",float(d_prod@d_high))
for label,direction in [("crown",d_crown),("prod",d_prod),("high",d_high)]:
    print("\n",label)
    for crit in ["low","medium","high","crown_jewel"]:
        vals=[]
        for priv in [False,True]:
            ix=[sec_cf_rows[(eid,crit,priv,"severity")] for eid in security_seed_ids]
            vals.append(np.round(np.mean((U14[ix][:,[1,3,4]]-U14[ix][:,[2]])@direction,axis=0),4).tolist())
        print(crit,"contrasts [1-2,3-2,4-2], unpriv/priv:",vals)
```
