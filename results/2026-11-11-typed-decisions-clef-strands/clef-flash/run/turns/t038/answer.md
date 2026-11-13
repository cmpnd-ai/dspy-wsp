I’ll measure how the edit changes responses to each controlled cue. This should separate genuine evidence sensitivity from simple shifts toward “partial,” “urgent,” or “false.”

```python
# Hypothesis: the bypass improves state-conditioned option margins, especially the response to tool errors and constraint violations.
pmeta=json.loads((WORK/"fwd_8/meta.json").read_text())["rows"]
P0={p.stem:torch.from_numpy(np.array(np.load(p))) for p in (WORK/"fwd_8").glob("*.npy")}
P1={p.stem:torch.from_numpy(np.array(np.load(p))) for p in (WORK/"fwd_9").glob("*.npy")}
pidx={(m["request"],m["question"]):i for i,m in enumerate(pmeta)}
def probe_margin(arr,request,q,pos,neg):
    i=pidx[request,q]
    opts=pmeta[i]["options"]
    return (arr["option_logits"][i,opts.index(pos)]-arr["option_logits"][i,opts.index(neg)]).item()
probe_summary=[]
for family,q,pos,neg in [
    ("agent_tool_errors","outcome","partial","success"),
    ("agent_constraint_violations","outcome","harmful","success"),
    ("agent_constraint_violations","urgency","3","1"),
    ("invoice_duplicate","duplicate","true","false"),
    ("security_benign_context","credential_compromise","true","false"),
    ("security_benign_context","true_positive","true","false")
]:
    levels=list(dict.fromkeys(m["level"] for m in probe_meta if m["family"]==family))
    print("\n",family,q,f"{pos} minus {neg}")
    for level in levels:
        inds=[i for i,m in enumerate(probe_meta) if m["family"]==family and m["level"]==level]
        vals=np.array([[probe_margin(a,i,q,pos,neg) for a in [P0,P1]] for i in inds])
        print(str(level),"original/edited margin",vals.mean(0).round(3),
              "correction",round(np.diff(vals,axis=1).mean(),3))
        probe_summary.append((family,q,str(level),vals.mean(0).tolist()))
print("\nFeatures unchanged on controlled probes:",
      torch.nan_to_num(P0["head_inputs"]-P1["head_inputs"]).abs().max().item())
note("## Passes 8–9 controlled cue response\n"+json.dumps(probe_summary)+
     "\nRead original and edited on identical state perturbations; scorer features unchanged. These probes are mechanistic evidence only, not scored accuracy estimates.")
```

This comparison changes no weights; the validation-confirmed edit remains saved.
