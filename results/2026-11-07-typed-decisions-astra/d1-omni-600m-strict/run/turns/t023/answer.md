I’ll measure the matched deadline effect and build a direction from those changes, rather than from wrong answers alone. I’ll remove fixed option-wording contrasts so the candidate targets payment evidence, not a blanket preference for “elevated.”
The analysis below checks the counterfactual logit response and isolates the payment-sensitive direction while protecting unrelated option distinctions.
```python
# Hypothesis: matched changes in payment deadlines define a state-sensitive direction that the current scorer underreads.
F6=WORK/"fwd_6"
meta6=json.loads((F6/"meta.json").read_text())
arr6={p.stem:np.load(p,mmap_mode="r") for p in F6.glob("*.npy")}
reqs6=json.loads((F6/"requests.json").read_text())
assert [canonical(r) for r in reqs6]==[canonical(r) for r in synthetic_requests]
U6=unit_center_features(np.asarray(arr6["head_inputs"]))
TU6=unit_center_features(np.asarray(arr6["trunk_markers"]))
syn_rows={}
for i,m in enumerate(meta6["rows"]):
    tag=synthetic_tags[m["request"]]
    syn_rows[tag["id"],tag["variant"],m["question"]]=i
for tag,_ in variants:
    ii=[syn_rows[eid,tag,"urgency"] for eid in invoice_seed_ids]
    logits=np.asarray(arr6["option_logits"])[ii]
    print(tag,"urgency argmax",collections.Counter(logits.argmax(-1).tolist()),
          "elevated-routine mean",round(float(np.mean(logits[:,2]-logits[:,1])),3),
          "critical-routine mean",round(float(np.mean(logits[:,3]-logits[:,1])),3))
# Double-difference isolates how deadline changes alter urgency option distinctions.
cf_vectors={}
for tag in ["near","overdue","discount"]:
    dd=[]
    for eid in invoice_seed_ids:
        i=syn_rows[eid,tag,"urgency"]; j=syn_rows[eid,"later","urgency"]
        dd.append((U6[i,2]-U6[i,1])-(U6[j,2]-U6[j,1]))
    cf_vectors[tag]=np.stack(dd)
    a,b=cf_vectors[tag][::2].mean(0),cf_vectors[tag][1::2].mean(0)
    print(tag,"contrast norm",np.linalg.norm(np.mean(dd,0)),"half cosine",a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
# Null fixed option contrasts from unrelated question families plus later-due invoice urgency.
nuisance=[]
for key in sorted(groups):
    if key==("invoice_processing","urgency"):
        ii=[syn_rows[eid,"later","urgency"] for eid in invoice_seed_ids]
        mat=np.mean(U6[ii],axis=0)
    else:
        rr=[r for r in rowinfo2 if (r["wf"],r["q"])==key]
        mat=np.mean([U2[r["i"],:len(r["opts"])] for r in rr],axis=0)
    for j in range(1,len(mat)):
        nuisance.append(mat[j]-mat[0])
nuisance.extend([eff_unit,np.ones(1024)/32])
Qn,_=np.linalg.qr(np.stack(nuisance,axis=1))
causal_mean=np.mean(np.concatenate(list(cf_vectors.values())),axis=0)
d_causal_raw=causal_mean-Qn@(Qn.T@causal_mean)
d_causal=d_causal_raw/np.linalg.norm(d_causal_raw)
print("Causal contrast norm before/after nuisance null",np.linalg.norm(causal_mean),np.linalg.norm(d_causal_raw),
      "null rank",Qn.shape[1])
for tag,_ in variants:
    ii=[syn_rows[eid,tag,"urgency"] for eid in invoice_seed_ids]
    print(tag,"causal (2-1) projection",float(np.mean([(U6[i,2]-U6[i,1])@d_causal for i in ii])))
for name,uf,info in [("A",U2,rowinfo2),("B",U3,rowinfo3)]:
    rs=[r for r in info if r["wf"]=="invoice_processing" and r["q"]=="urgency"]
    for flag in [False,True]:
        rr=[r for r in rs if (by_id[r["id"]]["inputs"]["state"]["payment"]["days_until_due"]<=3)==flag]
        print(name,"near",flag,"projection mean,sd",np.round([np.mean([(uf[r["i"],2]-uf[r["i"],1])@d_causal for r in rr]),
                                                            np.std([(uf[r["i"],2]-uf[r["i"],1])@d_causal for r in rr])],4))
print("Max other-question mean option span on B:")
other_spans=[]
for key in sorted(groups):
    rs=[r for r in rowinfo3 if (r["wf"],r["q"])==key]
    projections=[U3[r["i"],:len(r["opts"])]@d_causal for r in rs]
    other_spans.append((key,float(np.mean([np.ptp(p) for p in projections]))))
print(sorted(other_spans,key=lambda kv:-kv[1])[:8])
```
