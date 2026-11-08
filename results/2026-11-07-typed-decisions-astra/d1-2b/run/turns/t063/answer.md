The 720-record repair gains 14.56 points on the 180 records excluded from fitting, with a paired SE of 1.93 points. I’ll record the result and run the final validation check. If it falls below the saved best, I’ll restore that earlier candidate.
```python
# Hypothesis: the 720-record fixed-recipe repair retains a genuine held-out gain and merits final hidden validation.
records24 = [json.loads(s) for s in (work_dir/"gen_24_edited.jsonl").read_text().splitlines()]
records19_map = {r["id"]:r for r in records19}
delta720_vs540 = np.array([r["score"]-records19_map[r["id"]]["score"] for r in records24])
print("720 minus 540 on remaining 180:",
      float(delta720_vs540.mean()),
      "paired SE", float(delta720_vs540.std(ddof=1)/np.sqrt(len(delta720_vs540))))
for wf in sorted({r["inputs"]["workflow"] for r in records24}):
    rr = [r for r in records24 if r["inputs"]["workflow"]==wf]
    dd = np.array([r["score"]-base_by_id[r["id"]]["score"] for r in rr])
    print(wf, "n", len(rr), "gain %.4f SE %.4f" % (dd.mean(),dd.std(ddof=1)/np.sqrt(len(dd))))
with (work_dir/"notes.md").open("a") as f:
    f.write("\nPass 24 final720 coefficient-excluded practice check n=180: original .621111 -> edited .766667; paired gain +.145556 SE .019288; 99 improved, 29 worsened. Final720 minus expanded540 on the same records +%.6f SE %.6f. No further fitting or strength selection.\n" % (delta720_vs540.mean(),delta720_vs540.std(ddof=1)/np.sqrt(len(delta720_vs540))))
    f.write("\nPlanned pass 25 (final forward): DECIDE both hidden validation n=300. Exact current edit W2_original+B A_final720 W2_original, rank<=9, ridge1 alpha1, only layers.23.feed_forward.w2.weight changed. After the result, retain the highest-scoring validation-confirmed candidate; expanded540 remains saved in best_expanded540/ and first candidate in best_1/. No more candidate tuning.\n")
```