The extra protection reduced off-target variation only modestly. I’ll remove another specific nuisance: differences between alert narratives that remain when criticality and privileges are held fixed. This should help the readout respond to the intended state field rather than incidental wording.
```python
# Hypothesis: within-condition alert variation contaminates security cue directions; removing its leading PCs improves specificity.
security_within_samples=[]
for crit in ["low","medium","high","crown_jewel"]:
    for priv in [False,True]:
        ix=[sec_cf_rows[(eid,crit,priv,"severity")] for eid in security_seed_ids]
        x=U14[ix]
        x=x-x.mean(1,keepdims=True)
        x=x-x.mean(0,keepdims=True)
        security_within_samples.extend(x.reshape(-1,1024))
_,sec_within_sv,sec_within_vh=np.linalg.svd(np.stack(security_within_samples),full_matrices=False)
sec_within_pcs=list(sec_within_vh[:48])
d_crown_specific,cn_spec,_=null_project_direction(crit_diffs.mean(0),crown_ns128+sec_within_pcs)
d_prod_specific,pn_spec,_=null_project_direction(priv_diffs.mean(0),prod_ns128+sec_within_pcs)
d_high_specific,hn_spec,_=null_project_direction(high_diffs.mean(0),prod_ns128+sec_within_pcs)
d_low_specific,ln_spec,_=null_project_direction(low_diffs.mean(0),low_ns128+sec_within_pcs)
print("Specific residual norms:",np.round([cn_spec,pn_spec,hn_spec,ln_spec],5))
for label,direction,a,b in [("crown",d_crown_specific,4,2),("prod",d_prod_specific,3,2),
                            ("high",d_high_specific,3,2),("low",d_low_specific,1,2)]:
    print(label,"off-target RMS",round(off_rms(direction),5))
    for present in [False,True]:
        ix=[]
        for r in rowinfo3:
            if r["wf"]!="security_incidents" or r["q"]!="severity": continue
            st=by_id[r["id"]]["inputs"]["state"]
            crit=st["context"]["asset_criticality"]
            priv="deploy:production" in st["principal"]["privileges"]
            cue={"crown":crit=="crown_jewel","prod":priv,"high":crit=="high","low":crit=="low" and not priv}[label]
            if cue==present: ix.append(r["i"])
        vals=(U3[ix,a]-U3[ix,b])@direction
        print(" cue",present,"n",len(ix),"projection mean/sd",round(float(vals.mean()),4),round(float(vals.std()),4))
```
This is a geometry-only analysis of exported features; it does not run the network or change the weights.
