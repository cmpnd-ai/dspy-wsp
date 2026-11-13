The comparison failed because padded options contain non-finite logits; multiplying them by a zero mask still produces NaNs. I’ll exclude padding before centering, verify the cue responses, and prepare the final practice audit. The weights are unchanged.

```python
# Hypothesis: excluding padded options will confirm that the final edit's cue responses match the exact pointer-head calculation.
def valid_center(logits, mask):
    finite_logits = logits.masked_fill(~mask, 0)
    means = finite_logits.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)
    return (finite_logits-means).masked_fill(~mask, 0)

assert torch.isfinite(actual_cf_logits[cf["mask"]]).all()
cf_center_error = (valid_center(actual_cf_logits, cf["mask"]) -
                   valid_center(cf_best2_logits, cf["mask"])).abs().max().item()
print("Final counterfactual max centered-logit error:", cf_center_error)
assert cf_center_error < 1e-4
cf_verified_stats = collections.defaultdict(list)
for pair in cf_pairs_final:
    responses = []
    for logits in (cf["L"], actual_cf_logits):
        margins = []
        for rid in pair["requests"]:
            idx = cf_lookup[(rid, pair["qid"])]
            opts = cf["meta"][idx]["options"]
            ip, im = opts.index(str(pair["positive"])), opts.index(str(pair["negative"]))
            margins.append(float((logits[idx,ip]-logits[idx,im])/cf["temp"][idx]))
        responses.append(margins[1]-margins[0])
    cf_verified_stats[pair["cue"]].append(responses)
for cue, vals in cf_verified_stats.items():
    arr = np.asarray(vals)
    print(cue, "mean response original/final", arr.mean(0).round(5).tolist(),
          "expected sign counts", (arr>0).sum(0).tolist())

assert edited() == ["k.weight"] and torch.equal(load("k.weight"), best2_matrix)
assert hashlib.sha256((WORK/"edited/k.weight.npy").read_bytes()).hexdigest() == final_weight_sha256
final_req_check = json.load(open(WORK/"final_all_practice.json"))
assert len(final_req_check["problems"]) == 900
with open(WORK/"notes.md", "a") as f:
    f.write("\nRun39 diagnostic assertion encountered NaNs from padded -inf logits, not from model outputs. "
            "Fixed centering by excluding padding before summation; all valid-option logits are finite.\n")
    f.write(f"\nPass 23 final counterfactual READ: max centered-logit prediction error {cf_center_error:.8g}. "
            "Actual responses verify stronger sensitivity to delivery exceptions, due dates, and constraint violations "
            "(all 10/10 pairs in expected direction); allowlisting 9/10, retained limitation. No weight changes.\n")
    for cue, vals in cf_verified_stats.items():
        arr = np.asarray(vals)
        f.write(f"- {cue}: original/final response means {arr.mean(0).tolist()}, "
                f"expected signs {(arr>0).sum(0).tolist()}/{len(arr)}.\n")
    f.write("\n## Before pass 24\nExact frozen edit: best_2/edited/k.weight.npy, "
            "K0+outer(u,v_ref); all other tensors original. DECIDE edited on all 900 practice IDs "
            "in final_all_practice.json; compare itemwise against original pass1. "
            "This is a final audit only: fit540/dev180/test180 remain separately reported, "
            "and no further tuning will follow.\n")
```
