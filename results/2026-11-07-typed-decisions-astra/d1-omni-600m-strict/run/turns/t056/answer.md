I’ll compare matched severity margins and the geometry of the exported head inputs. The key test is whether criticality and privilege cues are clearly represented even when the original scorer keeps selecting medium severity.
```python
# Hypothesis: matched changes in criticality and privileges produce separable severity directions that the existing scorer underweights.
meta14 = json.loads((WORK/"fwd_14"/"meta.json").read_text())
arr14 = {p.stem:np.load(p,mmap_mode="r") for p in (WORK/"fwd_14").glob("*.npy")}
reqs14 = json.loads((WORK/"fwd_14"/"requests.json").read_text())
tagmap14 = {canonical(r):t for r,t in zip(security_cf_requests,security_cf_tags)}
tags14 = [tagmap14[canonical(r)] for r in reqs14]
U14 = unit_center_features(arr14["head_inputs"])
sec_cf_rows = {}
for i,m in enumerate(meta14["rows"]):
    t = tags14[m["request"]]
    sec_cf_rows[(t["seed"],t["criticality"],t["privileged"],m["question"])] = i
for crit in ["low","medium","high","crown_jewel"]:
    for priv in [False,True]:
        ix = [sec_cf_rows[(eid,crit,priv,"severity")] for eid in security_seed_ids]
        lg = arr14["option_logits"][ix]
        choices = np.argmax(lg,axis=1)
        print(crit,priv,"preds",collections.Counter(choices.tolist()),
              "mean gaps 1-2,3-2,4-2",np.round(np.mean(lg[:,[1,3,4]]-lg[:,[2]],axis=0),3).tolist())
# Each counterfactual difference compares option contrasts, removing shared record features.
def security_contrast(seed,crit,priv,a,b):
    i = sec_cf_rows[(seed,crit,priv,"severity")]
    return U14[i,a]-U14[i,b]
crit_diffs = np.stack([
    security_contrast(eid,"crown_jewel",priv,4,2)-security_contrast(eid,"medium",priv,4,2)
    for eid in security_seed_ids for priv in [False,True]])
priv_diffs = np.stack([
    security_contrast(eid,crit,True,3,2)-security_contrast(eid,crit,False,3,2)
    for eid in security_seed_ids for crit in ["low","medium","high"]])
high_diffs = np.stack([
    security_contrast(eid,"high",priv,3,2)-security_contrast(eid,"medium",priv,3,2)
    for eid in security_seed_ids for priv in [False,True]])
for label,x in [("crown",crit_diffs),("privilege",priv_diffs),("high_asset",high_diffs)]:
    half = x.shape[0]//2
    m1,m2 = x[:half].mean(0),x[half:].mean(0)
    print(label, "mean norm",round(float(np.linalg.norm(x.mean(0))),5),
          "half replication cosine",round(float(m1@m2/(np.linalg.norm(m1)*np.linalg.norm(m2))),3))
print("Direction cosine crown/priv:",float(crit_diffs.mean(0)@priv_diffs.mean(0)/(np.linalg.norm(crit_diffs.mean(0))*np.linalg.norm(priv_diffs.mean(0)))))
```
The validated checkpoint remains unchanged during this diagnostic.
