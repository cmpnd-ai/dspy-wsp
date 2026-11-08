The failing records suggest that asset criticality is underweighted: crown-jewel and high-criticality assets often still receive medium severity. I’ll check this with matched counterfactuals that change only asset criticality, keeping the alert evidence and account details fixed.
```python
# Hypothesis: asset-criticality information is represented in option states but insufficiently read by the severity scorer.
print("Security severity criteria:", by_id[security_a_ids[0]]["labels"]["questions"]["severity"])
for criticality in ["low","medium","high","crown_jewel"]:
    rr = [r for r in disc_rows if r["wf"]=="security_incidents" and r["q"]=="severity"
          and by_id[r["id"]]["inputs"]["state"]["context"]["asset_criticality"]==criticality]
    print(criticality, "n",len(rr),"accuracy",round(np.mean([r["ok"] for r in rr]),3),
          "gold",collections.Counter(r["gold"] for r in rr),"pred",collections.Counter(r["pred"] for r in rr))
print("Original request sample:",json.dumps(json.loads((WORK/"fwd_2"/"requests.json").read_text())[0])[:1500])
print("Projection helper metadata:")
for fname in ["protected_directions","install_linear_cues"]:
    val = globals()[fname]
    if callable(val):
        print(fname, inspect.signature(val), val.__code__.co_varnames, val.__code__.co_names)
    else:
        print(fname, type(val), getattr(val,"shape",None))
print("Feature array shapes:", {k:np.shape(globals()[k]) for k in ["U2","U3","U6","U9","d_payment","d_violation2"]})
```
This inspection uses discovery cases only. The validated checkpoint remains unchanged.
