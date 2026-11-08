The higher dose overshot: it strengthened the intended production-access margin but increased unrelated errors. I’ll reject it and improve the directions’ specificity by removing more variation from unrelated option contrasts. I’ll also check whether the existing counterfactuals support a separate low-impact cue, since the original model almost never selects low severity.
The higher dose overshot: it strengthened the intended production-access margin but increased unrelated errors. I’ll reject it and remove more unrelated variation from the cue directions. I’ll also check for a low-impact direction, since the original model almost never selects low severity.
```python
# Hypothesis: stronger nuisance projection will preserve security cue signals while reducing collateral option shifts.
note("\nPass16 high security dose: discovery B .581 vs original .557, +.0240 SE .01191; increment vs base repair -.0070 SE .00923. Severity .46 versus .50 low dose. Production margin increases monotonically -.790,+.092,+.973, but high dose creates collateral errors. Reject high; refine specificity using discovery-A nuisance geometry only.")
# Build label-free nuisance variation from unrelated questions in discovery A and matched controls.
sec_nuisance_samples=list(nonsec_contrasts)
for r in rowinfo2:
    if r["wf"]=="security_incidents" and r["q"]!="severity":
        x=U2[r["i"],:len(r["opts"])]
        sec_nuisance_samples.extend(x-x.mean(0))
for i,m in enumerate(meta14["rows"]):
    if m["question"]!="severity":
        x=U14[i,:len(m["options"])]
        sec_nuisance_samples.extend(x-x.mean(0))
_, sec_sv128, sec_vh128 = np.linalg.svd(np.stack(sec_nuisance_samples),full_matrices=False)
sec_common128=list(nonsec_means)+list(sec_vh128[:128])
for crit in ["low","medium","high","crown_jewel"]:
    for priv in [False,True]:
        for qid in ["credential_compromise","disposition","true_positive","urgency"]:
            ix=[sec_cf_rows[(eid,crit,priv,qid)] for eid in security_seed_ids]
            nopt=len(meta14["rows"][ix[0]]["options"])
            sec_common128.extend(U14[ix,:nopt].mean(0))
crown_ns128=list(sec_common128)
for crit in ["low","medium","high"]:
    for priv in [False,True]:
        ix=[sec_cf_rows[(eid,crit,priv,"severity")] for eid in security_seed_ids]
        crown_ns128.extend(U14[ix].mean(0))
prod_ns128=list(sec_common128)
for crit in ["low","medium"]:
    ix=[sec_cf_rows[(eid,crit,False,"severity")] for eid in security_seed_ids]
    prod_ns128.extend(U14[ix].mean(0))
for priv in [False,True]:
    ix=[sec_cf_rows[(eid,"crown_jewel",priv,"severity")] for eid in security_seed_ids]
    prod_ns128.append((U14[ix,4]-U14[ix,3]).mean(0))
d_crown128, cn128, _ = null_project_direction(crit_diffs.mean(0),crown_ns128)
d_prod128, pn128, _ = null_project_direction(priv_diffs.mean(0),prod_ns128)
d_high128, hn128, _ = null_project_direction(high_diffs.mean(0),prod_ns128)
low_diffs=np.stack([security_contrast(eid,"low",False,1,2)-security_contrast(eid,"medium",False,1,2)
                    for eid in security_seed_ids])
low_ns128=list(sec_common128)
for crit in ["medium","high","crown_jewel"]:
    for priv in [False,True]:
        ix=[sec_cf_rows[(eid,crit,priv,"severity")] for eid in security_seed_ids]
        low_ns128.extend(U14[ix].mean(0))
ix=[sec_cf_rows[(eid,"low",True,"severity")] for eid in security_seed_ids]
low_ns128.extend(U14[ix].mean(0))
d_low128, ln128, _ = null_project_direction(low_diffs.mean(0),low_ns128)
print("Protected norms crown/prod/high/low:",np.round([cn128,pn128,hn128,ln128],5))
for label,d0,d1,a,b in [("crown",d_crown,d_crown128,4,2),("prod",d_prod,d_prod128,3,2),
                        ("high",d_high,d_high128,3,2),("low",None,d_low128,1,2)]:
    off_ix=[r["i"] for r in rowinfo3 if r["wf"]!="security_incidents" or r["q"]!="severity"]
    def off_rms(direction):
        vals=[]
        for i in off_ix:
            x=U3[i,:len(rowinfo3[i]["opts"])]
            vals.extend((x-x.mean(0))@direction)
        return float(np.sqrt(np.mean(np.square(vals))))
    print(label,"off-target projection RMS old/new",None if d0 is None else round(off_rms(d0),5),round(off_rms(d1),5))
    for present in [False,True]:
        ix=[]
        for r in rowinfo3:
            if r["wf"]!="security_incidents" or r["q"]!="severity": continue
            st=by_id[r["id"]]["inputs"]["state"]
            crit=st["context"]["asset_criticality"]
            priv="deploy:production" in st["principal"]["privileges"]
            cue={"crown":crit=="crown_jewel","prod":priv,"high":crit=="high","low":crit=="low" and not priv}[label]
            if cue==present: ix.append(r["i"])
        vals=(U3[ix,a]-U3[ix,b])@d1
        print("  cue",present,"n",len(ix),"projection mean/sd",round(float(vals.mean()),4),round(float(vals.std()),4))
```
