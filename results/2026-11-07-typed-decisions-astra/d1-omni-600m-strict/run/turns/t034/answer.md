I’ll inspect the counterfactual responses and isolate a violation-sensitive direction shared by outcome, risk, and urgency. I’ll protect the clean-state option contrasts and unrelated workflows, so the edit must respond to the violation cue rather than always favoring “harmful” or “high.”
```python
# Hypothesis: a constraint-violation cue survives in head features but is not read strongly enough for harmful/high/critical judgments.
F9=WORK/"fwd_9"
meta9=json.loads((F9/"meta.json").read_text())
arr9={p.stem:np.load(p,mmap_mode="r") for p in F9.glob("*.npy")}
reqs9=json.loads((F9/"requests.json").read_text())
assert [canonical(r) for r in reqs9]==[canonical(r) for r in agent_requests]
U9=unit_center_features(np.asarray(arr9["head_inputs"]))
agent_rows={}
for i,m in enumerate(meta9["rows"]):
    tag=agent_tags[m["request"]]
    agent_rows[tag["id"],tag["variant"],m["question"]]=i
for tag,_,_ in agent_variants:
    print("\n",tag)
    for q in ["outcome","risk","urgency"]:
        ii=[agent_rows[eid,tag,q] for eid in agent_seed_ids]
        mm=meta9["rows"][ii[0]]
        logits=np.asarray(arr9["option_logits"])[ii]
        print(q,dict(collections.Counter(mm["options"][j] for j in logits.argmax(-1))),
              "mean logits",np.round(logits.mean(0),2))
# Protect unrelated fixed option distinctions and clean agent distinctions.
agent_nuisance=[]
for key in sorted(groups):
    if key[0]=="agent_trace_observability":
        ii=[agent_rows[eid,"clean",key[1]] for eid in agent_seed_ids]
        n=len(meta9["rows"][ii[0]]["options"])
        mat=np.mean(U9[ii,:n],axis=0)
    else:
        rs=[r for r in rowinfo2 if (r["wf"],r["q"])==key]
        mat=np.mean([U2[r["i"],:len(r["opts"])] for r in rs],axis=0)
    agent_nuisance.extend(mat[1:]-mat[0])
agent_nuisance.extend([eff_unit,np.ones(1024)/32,d_payment])
Qa,_=np.linalg.qr(np.stack(agent_nuisance,axis=1))
agent_ns=[]
for r in rowinfo2:
    if r["wf"]!="agent_trace_observability":
        mat=U2[r["i"],:len(r["opts"])]
        agent_ns.extend(mat-mat.mean(0))
AN=np.stack(agent_ns).astype(np.float64)
AN-= (AN@Qa)@Qa.T
_,san,van=np.linalg.svd(AN,full_matrices=False)
Qap=np.concatenate([Qa,van[:32].T],axis=1)
violation_diffs=[]
for eid in agent_seed_ids:
    for tag in ["violation1","violation3","uncontrolled"]:
        for q,hi,lo in [("outcome",1,2),("risk",3,0),("urgency",3,1)]:
            i=agent_rows[eid,tag,q];j=agent_rows[eid,"clean",q]
            violation_diffs.append((U9[i,hi]-U9[i,lo])-(U9[j,hi]-U9[j,lo]))
vmean=np.mean(violation_diffs,axis=0)
vraw=vmean-Qap@(Qap.T@vmean)
d_violation=(vraw/np.linalg.norm(vraw)).astype(np.float32)
print("\nViolation contrast norm/raw after protection:",np.linalg.norm(vmean),np.linalg.norm(vraw))
for tag,_,_ in agent_variants:
    print(tag,{q:round(float(np.mean([(U9[agent_rows[eid,tag,q],hi]-U9[agent_rows[eid,tag,q],lo])@d_violation for eid in agent_seed_ids])),4)
               for q,hi,lo in [("outcome",1,2),("risk",3,0),("urgency",3,1)]})
for flag in [False,True]:
    rr=[r for r in rowinfo3 if r["wf"]=="agent_trace_observability" and
        (by_id[r["id"]]["inputs"]["state"]["trace_summary"]["constraint_violations"]>0)==flag]
    print("B violations",flag,[(q,round(float(np.mean([(U3[r["i"],hi]-U3[r["i"],lo])@d_violation for r in rr if r["q"]==q])),4))
                             for q,hi,lo in [("outcome",1,2),("risk",3,0),("urgency",3,1)]])
np.save(WORK/"violation_direction.npy",d_violation)
```
