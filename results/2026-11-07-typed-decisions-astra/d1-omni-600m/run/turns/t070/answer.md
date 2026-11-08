I’ll calculate the expected robustness results from the returned scorer inputs, then replay the same requests with the restored full repair. This is a diagnostic only; the edit remains frozen.
```python
# Hypothesis: adding irrelevant record metadata should leave the repair's evidence-dependent gain largely intact.
m23=json.load(open(W/"fwd_23"/"meta.json"))
mask23=torch.tensor(np.load(W/"fwd_23"/"option_mask.npy"))
p23_original=torch.tensor(np.nan_to_num(np.load(W/"fwd_23"/"probabilities.npy")))
input23=torch.tensor(np.nan_to_num(np.load(W/"fwd_23"/"head_inputs.npy")),dtype=torch.float32)
x23=F.layer_norm(input23,(1024,),scorer_params["0.weight"],scorer_params["0.bias"])
temp23=torch.tensor([cfg["temperatures"]["noul:2" if r["type"]=="noul" else (r["type"]+":3-5")] for r in m23["rows"]])
l23_predicted=logits_scorer_params(x23,best2_params)
p23_predicted=torch.softmax((l23_predicted/temp23[:,None]).masked_fill(~mask23,-1e6),-1)
gold23=torch.zeros_like(p23_original)
for j,row in enumerate(m23["rows"]):
    tag=robust_tags[row["request"]]
    gprob=by_id[tag["id"]]["labels"]["gold"][row["question"]]["probabilities"]
    for oi,opt in enumerate(row["options"]):
        gold23[j,oi]=gprob[str(opt)]
variant23=torch.tensor([robust_tags[r["request"]]["variant"]=="irrelevant_reference" for r in m23["rows"]])
def robustness_summary(probs):
    pred=probs.masked_fill(~mask23,-1e6).argmax(-1)
    correct=(pred==gold23.argmax(-1))
    unchanged=~variant23
    return {"original_state_accuracy":float(correct[unchanged].float().mean()),
            "metadata_added_accuracy":float(correct[variant23].float().mean()),
            "answer_flip_rate":float((pred[unchanged]!=pred[variant23]).float().mean()),
            "mean_probability_total_variation":float((probs[unchanged]-probs[variant23]).abs().sum(-1).mean()/2)}
print("Original robustness:",robustness_summary(p23_original))
print("Frozen repair predicted robustness:",robustness_summary(p23_predicted))
note("\nPass23 metadata robustness baseline read obtained. Original="+str(robustness_summary(p23_original))+"; frozen best2 local prediction="+str(robustness_summary(p23_predicted))+".\nNext pass24: READ edited identical200 requests/1000questions in metadata_robustness_edited.json. Exact installed edit conditional_candidate.pt/best2:110 scorer neurons modified (shared pair33/746 and54 conditional paired channels), no changes since restoration beforepass23. No fitting or selection; check actual robustness and scorer replay.")
```
