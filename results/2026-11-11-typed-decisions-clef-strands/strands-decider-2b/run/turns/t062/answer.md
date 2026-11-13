The full-strength run confirms the dose response: development accuracy is 57.78%, 69.78%, 77.22%, and 74.67% at 0×, 0.5×, 1×, and 1.5×. I’ll now check whether the final repair responds appropriately to controlled changes in delivery status, payment deadlines, constraint violations, and allowlisting.


```python
# Hypothesis: the final repair strengthens responses to relevant state cues rather than merely favoring fixed options.
cf_pairs_final = json.load(open(WORK/"counterfactual_pairs.json"))
print("Pair container:", type(cf_pairs_final).__name__, "count:", len(cf_pairs_final))
print("First pair:", cf_pairs_final[0])
print("First counterfactual metadata:", cf["meta"][:2])
cf_best2_logits = cf["L"] + correction(cf, v_ref)
cf_lookup = {(m["request"], m["question"]): i for i,m in enumerate(cf["meta"])}
cf_final_stats = collections.defaultdict(list)
for pair in cf_pairs_final:
    pair_margins = []
    for logits in (cf["L"], cf_best2_logits):
        margins = []
        for rid in pair["requests"]:
            row_idx = cf_lookup[(rid, pair["qid"])]
            options = cf["meta"][row_idx]["options"]
            pos = options.index(str(pair["positive"]))
            neg = options.index(str(pair["negative"]))
            margins.append(float((logits[row_idx,pos]-logits[row_idx,neg])/cf["temp"][row_idx]))
        pair_margins.append(margins[1]-margins[0])
    cf_final_stats[pair["cue"]].append(pair_margins)
for cue, vals in cf_final_stats.items():
    arr = np.array(vals)
    print(cue, "original/final mean signed log-odds response:", arr.mean(0).round(4).tolist(),
          "positive pairs:", (arr>0).sum(0).tolist(), "/", len(arr))
assert torch.equal(load("k.weight"), best2_matrix)
with open(WORK/"notes.md", "a") as f:
    f.write("\nPass 22 final full-strength development: original .577778 -> edited .772222, "
            "gain .194444, paired SE .018540; 115 improved, 21 worsened. "
            "Actual fixed-direction dose response 0/.5/1/1.5 = .577778/.697778/.772222/.746667.\n")
    for cue, vals in cf_final_stats.items():
        arr = np.array(vals)
        f.write(f"Final-edit predicted cue response {cue}: original/final means {arr.mean(0).tolist()}, "
                f"positive pair counts {(arr>0).sum(0).tolist()}/{len(arr)}.\n")
    f.write("\n## Before pass 23\nExact frozen edit: best2 k.weight = K0+outer(u,v_ref); "
            "all other tensors original. READ edited on the same 80 single-cue counterfactual requests "
            "(400 questions) in counterfactual_edited.json, paired definitions in counterfactual_pairs.json. "
            "All derived from fit examples, no synthetic labels used for fitting. "
            "Compare with pass11 original head inputs; verify final edit's causal cue responses.\n")
```
