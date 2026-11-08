I’ll check three things on the reserved examples: whether the option gates activate selectively, whether replacing the low-contribution neurons removed meaningful original signal, and whether local scorer replay matches the actual edited answers.
```python
# Hypothesis: the repair's out-of-sample effect is localized to selective residual channels, not accidental removal of the original scorer.
m20=json.load(open(W/"fwd_20"/"meta.json"))
print("Read metadata first rows",m20["rows"][:2])
print("Test record order",[r["id"] for r in test_orig[:4]])
t20_input=torch.tensor(np.nan_to_num(np.load(W/"fwd_20"/"head_inputs.npy")),dtype=torch.float32)
t20_x=F.layer_norm(t20_input,(1024,),scorer_params["0.weight"],scorer_params["0.bias"])
t20_mask=torch.tensor(np.load(W/"fwd_20"/"option_mask.npy"))
t20_base=torch.tensor(np.nan_to_num(np.load(W/"fwd_20"/"option_logits.npy")))
t20_full=logits_scorer_params(t20_x,best2_params)
# from-filter ordering follows the recorded decide file, not the requested list.
t20_ids=[r["id"] for r in test_orig if r["id"] in set(test_read_ids)]
t20_qid=torch.tensor([qkeys.index((by_id[t20_ids[r["request"]]]["inputs"]["workflow"],r["question"])) for r in m20["rows"]])
t20_gates=t20_x@idgateW+idmean-gate_thresholds
gate_pos=[]
gate_neg=[]
for qno,oi in selected_gate_keys:
    gi=qopt_index[(qno,oi)]
    gt=torch.zeros_like(t20_mask)
    gt[t20_qid==qno,oi]=True
    gate_pos.append(t20_gates[:,:,gi][gt])
    gate_neg.append(t20_gates[:,:,gi][t20_mask&~gt])
gate_pos=torch.cat(gate_pos);gate_neg=torch.cat(gate_neg)
print("Gate detection true positive rate",float((gate_pos>0).float().mean()),
      "false positive rate",float((gate_neg>0).float().mean()),
      "positive p05",float(torch.quantile(gate_pos,.05)),
      "negative p99",float(torch.quantile(gate_neg,.99)))
t20_z=F.gelu(F.linear(t20_x,scorer_params["1.weight"],scorer_params["1.bias"]))
all_modified_units=pair_units+cond_units
removed20=t20_z[:,:,all_modified_units]@scorer_params["3.weight"][0,all_modified_units]
def option_center(v,mask):
    return (v-(v*mask).sum(-1,keepdim=True)/mask.sum(-1,keepdim=True))*mask
print("Displaced original-neuron logit-contrast RMS",float(option_center(removed20,t20_mask).square().sum().div(t20_mask.sum()).sqrt()))
test_probs={i:probs_record(te[i]) for i in t20_ids}
t20_temp=torch.tensor([cfg["temperatures"]["noul:2" if r["type"]=="noul" else (r["type"]+":3-5")] for r in m20["rows"]])
t20_pp=torch.softmax((t20_full/t20_temp[:,None]).masked_fill(~t20_mask,-1e6),-1)
prob_errors=[]
for j,row in enumerate(m20["rows"]):
    iid=t20_ids[row["request"]]
    actual=test_probs[iid][row["question"]]
    for oi,opt in enumerate(row["options"]):
        prob_errors.append(abs(float(t20_pp[j,oi])-actual[str(opt)]))
print("Actual best2 vs local scorer max probability error",max(prob_errors))
note("\nPass20 frozen test read audit: first200 reserved items. Gate selectivity and original displaced-neuron contribution checked locally; no refitting. Best2 remains installed. Full test gain positive in all workflows: agent+24.21pp SE3.64; customer+9.47pp SE3.21; invoice+15.44pp SE3.17; security+7.02pp SE2.76.")
```
