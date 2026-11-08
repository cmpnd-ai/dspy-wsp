The expanded fit improves practice-test accuracy to 77.17%, a gain of 16.39 points over the original (paired SE 1.37 points). I’ll record the result and prepare its validation check, while retaining both confirmed candidates.
```python
# Hypothesis: the expanded fit's practice gain will carry over to hidden validation without changing the mechanism.
records19 = [json.loads(s) for s in (work_dir/"gen_19_edited.jsonl").read_text().splitlines()]
records14_map = {r["id"]:r for r in records14}
expanded_vs_first = np.array([r["score"]-records14_map[r["id"]]["score"] for r in records19])
print("Expanded minus first candidate, practice test:",
      float(expanded_vs_first.mean()),
      "paired SE", float(expanded_vs_first.std(ddof=1)/np.sqrt(len(expanded_vs_first))))
print("call_probs signature:", inspect.signature(call_probs))
print("Call structure:", list(records19[0]["calls"][0].keys()))
print("Completion sample:", str(records19[0]["calls"][0].get("completion"))[:2000])
with (work_dir/"notes.md").open("a") as f:
    f.write("\nPass 19 expanded540 reserved practice test n=360: original .607778 -> edited .771667; paired gain +.163889 SE .013709; 212 improved, 50 worsened. Expanded minus first candidate +%.6f, paired SE %.6f (not enough to establish superiority, but both beat original).\n" % (expanded_vs_first.mean(),expanded_vs_first.std(ddof=1)/np.sqrt(len(expanded_vs_first))))
    f.write("\nPlanned pass 20: DECIDE both hidden validation n=300. Exact current edit W2_original+B A_expanded540 W2_original, rank<=9, ridge1, alpha1, only layers.23.feed_forward.w2.weight changed. Final candidate coefficients use practice fit+dev only; no validation labels accessible.\n")
```