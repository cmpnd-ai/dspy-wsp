The counterfactuals confirm that violations move the internal representation, yet all 16 matched violation cases still choose “partial.” The first direction also reacts to ordinary tool errors, so I’ll refine it using violation-versus-irreversible pairs where all other counters are identical. I’ll also inspect separate directions for irreversible actions and clean completion.
```python
# Hypothesis: isolating each counter with matched controls prevents a violation repair from misclassifying routine errors as harmful.
def agent_cue_direction(targets, variant_pairs, protected_variants, protect_other_agent_questions=True):
    # targets = (question, preferred-option-index, competing-option-index) for a specific semantic cue.
    target_qs={q for q,_,_ in targets}
    nuisance=[]
    for key in sorted(groups):
        if key[0]=="agent_trace_observability":
            variants_to_protect=protected_variants if key[1] in target_qs else [v[0] for v in agent_variants]
            if not protect_other_agent_questions and key[1] not in target_qs:
                variants_to_protect=["clean"]
            for tag in variants_to_protect:
                ii=[agent_rows[eid,tag,key[1]] for eid in agent_seed_ids]
                n=len(meta9["rows"][ii[0]]["options"])
                mat=np.mean(U9[ii,:n],axis=0)
                nuisance.extend(mat[1:]-mat[0])
        else:
            rr=[r for r in rowinfo2 if (r["wf"],r["q"])==key]
            mat=np.mean([U2[r["i"],:len(r["opts"])] for r in rr],axis=0)
            nuisance.extend(mat[1:]-mat[0])
    nuisance.extend([eff_unit,np.ones(1024)/32,d_payment])
    Q,_=np.linalg.qr(np.stack(nuisance,axis=1))
    samples=[]
    for r in rowinfo2:
        if r["wf"]!="agent_trace_observability":
            mat=U2[r["i"],:len(r["opts"])]
            samples.extend(mat-mat.mean(0))
    samples=np.stack(samples).astype(np.float64)
    samples-=(samples@Q)@Q.T
    _,_,v=np.linalg.svd(samples,full_matrices=False)
    Q=np.concatenate([Q,v[:32].T],axis=1)
    dd=[]
    for eid in agent_seed_ids:
        for positive,negative in variant_pairs:
            for q,hi,lo in targets:
                i=agent_rows[eid,positive,q];j=agent_rows[eid,negative,q]
                dd.append((U9[i,hi]-U9[i,lo])-(U9[j,hi]-U9[j,lo]))
    mean=np.mean(dd,axis=0)
    residual=mean-Q@(Q.T@mean)
    direction=(residual/np.linalg.norm(residual)).astype(np.float32)
    return direction,{"contrast_norm":float(np.linalg.norm(mean)),"residual_norm":float(np.linalg.norm(residual)),"protected_rank":Q.shape[1]}
d_violation2,vstats=agent_cue_direction(
    [("outcome",1,2),("risk",3,0),("urgency",3,1)],
    [("violation1","irreversible"),("violation3","irreversible")],
    ["clean","dry","error1","error4","irreversible"])
print("Refined violation",vstats)
for tag,_,_ in agent_variants:
    print(tag,{q:round(float(np.mean([(U9[agent_rows[eid,tag,q],hi]-U9[agent_rows[eid,tag,q],lo])@d_violation2 for eid in agent_seed_ids])),4)
               for q,hi,lo in [("outcome",1,2),("risk",3,0),("urgency",3,1)]})
for flag in [False,True]:
    rr=[r for r in rowinfo3 if r["wf"]=="agent_trace_observability" and
        (by_id[r["id"]]["inputs"]["state"]["trace_summary"]["constraint_violations"]>0)==flag]
    print("B violations",flag,[(q,round(float(np.mean([(U3[r["i"],hi]-U3[r["i"],lo])@d_violation2 for r in rr if r["q"]==q])),4))
                             for q,hi,lo in [("outcome",1,2),("risk",3,0),("urgency",3,1)]])
print("Unrelated B mean span",float(np.mean([np.ptp(U3[r["i"],:len(r["opts"])]@d_violation2) for r in rowinfo3 if r["wf"]!="agent_trace_observability"])))
np.save(WORK/"violation_direction_refined.npy",d_violation2)
```
I’ll use the refined direction only if it separates actual violations from routine errors and irreversible actions without violations.
