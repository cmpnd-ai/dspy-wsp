I’ll compare the two-neuron repair and the full extension on the same frozen probe features, then verify those predictions with actual forward calls. This isolates the extension’s effect from both the backbone and the shared repair.

```python
# Hypothesis: the gated extension changes responses to its three target fields while leaving upstream evidence and unrelated fields intact.
def read_probe_arrays(passno):
    folder=WORK/f"fwd_{passno}"
    mm=json.load(open(folder/"meta.json"))["rows"]
    ff=torch.nan_to_num(torch.from_numpy(np.load(folder/"head_inputs.npy")))
    mk=torch.from_numpy(np.load(folder/"option_mask.npy"))
    ll=torch.nan_to_num(torch.from_numpy(np.load(folder/"option_logits.npy"))).masked_fill(~mk,-torch.inf)
    return mm,ff,mk,ll
def simulate_probe_scoring(features,mask,original_logits,tensors):
    rw,rb,ro=tensors
    ro=ro.flatten()
    ch=((rw!=repair_W0).any(1)|(rb!=repair_b0)|(ro!=repair_wout)).nonzero().flatten()
    an=F.gelu(F.linear(features,rw[ch],rb[ch]))
    ao=F.gelu(F.linear(features,repair_W0[ch],repair_b0[ch]))
    return (original_logits+repair_gate*((an*ro[ch]).sum(-1)-(ao*repair_wout[ch]).sum(-1))).masked_fill(~mask,-torch.inf)
target_mm,target_ff,target_mk,target_l0=read_probe_arrays(16)
target_l1=simulate_probe_scoring(target_ff,target_mk,target_l0,repair_base1_tensors)
target_l2=simulate_probe_scoring(target_ff,target_mk,target_l0,repair_b2["tensors"])
target_summary=collections.defaultdict(list)
for ri,mm in enumerate(target_mm):
    pm=target_probe_meta[mm["request"]]
    q=mm["question"]
    options=mm["options"]
    comparisons=[]
    if pm["family"].startswith("urgency") and q=="urgency": comparisons=[("3","1")]
    if pm["family"].startswith("customer") and q=="needs_human": comparisons=[("true","false")]
    if pm["family"].startswith("customer") and q=="action":
        plus={"routine_information":"answer_directly","human_authority":"escalate_to_human","resolved":"close_no_action","verified_refund":"execute_refund","missing_information":"request_information"}[pm["level"]]
        comparisons=[(plus,"best_other")]
    for plus,minus in comparisons:
        jp=options.index(plus)
        vals=[]
        for ll in [target_l0,target_l1,target_l2]:
            mn=ll[ri,options.index(minus)] if minus!="best_other" else max(float(ll[ri,j]) for j in range(len(options)) if j!=jp)
            vals.append(float(ll[ri,jp]-mn))
        target_summary[(pm["family"],str(pm["level"]),q,plus+"-"+minus)].append(vals)
target_summary_table=[]
for key,vals in target_summary.items():
    item=(key,np.round(np.mean(vals,axis=0),3).tolist())
    target_summary_table.append(item)
    print(item)
non_target_rows=torch.tensor([not ((target_probe_meta[m["request"]]["family"].startswith("urgency") and m["question"]=="urgency") or
                                 (target_probe_meta[m["request"]]["family"].startswith("customer") and m["question"] in ["action","needs_human"])) for m in target_mm])
target_leak=center_options(torch.nan_to_num(target_l2-target_l1),target_mk)
print("Extension centered effect RMS unrelated fields:",float(target_leak[non_target_rows][target_mk[non_target_rows]].square().mean().sqrt()))
for name,arr in zip(["residual_scorer.0.weight","residual_scorer.0.bias","residual_scorer.3.weight"],repair_base1_tensors):
    save_edit(name,arr.reshape(SHAPES[name]))
with open(WORK/"notes.md","a") as f:
    f.write("\n## Pass 16 local target-cue analysis\nSimulated original/shared/best_2 target-cue margins: "+str(target_summary_table)+". Unscored probes only.\n")
    f.write("\n## Planned pass 17\nExact edit: best_1 two-neuron repair from linear_bypass_candidate.pt; no gated neurons. READ edited on same 156 target-cue requests (target_cue_probes_edited.json). Will compare exact features/logits with pass16 simulation. This is causal ablation of 22 gated neurons, not a candidate-selection run.\n")
```
