The refined cues improved severity, but their linear readouts still disturbed unrelated questions. I’ll test a more localized mechanism: use the scorer’s existing GELU nonlinearity to activate the security corrections only for the security-severity question. First I’ll look for a question-identifying direction that is stable across alert content and options.
```python
# Hypothesis: collateral errors arise because cue readouts are globally linear; a question-specific activation gate can localize them.
note("\nPass17 refined security moderate: discovery B .577, gain +.0200 SE .01236 vs original, -.0110 SE .00932 vs validated base. Severity .52 but substantial unrelated errors; reject as global linear add-on. Next test localized nonlinear gate using question-identifying feature direction.")
target_means=[]
gate_other_means=[]
gate_within=[]
for wf,qid in sorted({(r["wf"],r["q"]) for r in rowinfo2}):
    ix=[r["i"] for r in rowinfo2 if r["wf"]==wf and r["q"]==qid]
    no=len(rowinfo2[ix[0]]["opts"])
    x=U2[ix,:no]
    if wf=="security_incidents" and qid=="severity":
        target_means.append(x.mean((0,1)))
        target_opt_mean=x.mean(0)
    else:
        gate_other_means.extend(x.mean(0))
    # Remove within-question content variation, not differences between questions.
    gate_within.extend((x-x.mean(0,keepdims=True)).reshape(-1,1024))
_,gate_sv,gate_vh=np.linalg.svd(np.stack(gate_within),full_matrices=False)
target_cf_ix=[i for i,m in enumerate(meta14["rows"]) if m["question"]=="severity"]
target_mean=np.mean(U14[target_cf_ix],axis=(0,1))
gate_nuisance=list(gate_other_means)+list(target_opt_mean-target_opt_mean.mean(0))+list(gate_vh[:64])
for crit in ["low","medium","high","crown_jewel"]:
    for priv in [False,True]:
        ix=[sec_cf_rows[(eid,crit,priv,"severity")] for eid in security_seed_ids]
        # Keep gate independent of the controlled cue conditions and option identity.
        gate_nuisance.extend(U14[ix].mean(0)-target_mean)
d_security_gate,gate_resnorm,gate_rank=null_project_direction(target_mean,gate_nuisance)
print("Gate residual norm/rank:",gate_resnorm,gate_rank)
gate_projection={}
for label,ar,info in [("A",U2,rowinfo2),("B",U3,rowinfo3)]:
    for target in [False,True]:
        vals=np.concatenate([ar[r["i"],:len(r["opts"])]@d_security_gate for r in info
                             if (r["wf"]=="security_incidents" and r["q"]=="severity")==target])
        gate_projection[(label,target)]=vals
        print(label,"target",target,"n",len(vals),"projection quantiles",
              np.round(np.quantile(vals,[0,.01,.1,.5,.9,.99,1]),4).tolist())
np.save(WORK/"security_question_gate.npy",d_security_gate)
```
