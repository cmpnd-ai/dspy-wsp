The final repair increases sensitivity to the tested cues, though one allowlist pair remains a limitation. I’ll verify the readout numerically, then run the frozen edit on all 900 practice records for a complete audit—not for further tuning.


```python
# Hypothesis: the frozen rank-one repair's observed cue responses agree with its exact pointer-head mechanism.
actual_cf_logits = torch.from_numpy(np.load(WORK/"fwd_23/option_logits.npy")).float()
actual_cf_mask = torch.from_numpy(np.load(WORK/"fwd_23/option_mask.npy")).bool()
actual_cf_meta = json.load(open(WORK/"fwd_23/meta.json"))
assert torch.equal(actual_cf_mask, cf["mask"])
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
with open(WORK/"notes.md", "a") as f:
    f.write(f"\nPass 23 final counterfactual READ: max centered-logit prediction error {cf_center_error:.8g}. "
            "Actual responses verify stronger sensitivity to delivery exceptions, due dates, and constraint violations "
            "(all 10/10 pairs in expected direction); allowlisting 9/10, retained limitation. No weight changes.\n")
    f.write("\n## Before pass 24\nExact frozen edit: best_2/edited/k.weight.npy, "
            "K0+outer(u,v_ref); all other tensors original. DECIDE edited on all 900 practice IDs "
            "in final_all_practice.json; compare itemwise against original pass1. "
            "This is a final audit only: fit540/dev180/test180 remain separately reported, "
            "and no further tuning will follow.\n")
```
