The linear-target edit gives a clear gain over the original model on reserved practice, but is effectively tied with the earlier repair. I’ll validate it before making the final choice; the small difference between the two edits should not be treated as established.


```python
# Hypothesis: the linear-target rank-one repair may retain the earlier repair's validation gain while improving probability fit.
r18 = read_jsonl(WORK/"gen_18_edited.jsonl")
s18 = {r["id"]:r["score"] for r in r18}
inc18 = np.array([s18[k]-a13[k] for k in a13])
print("Linear minus best2 reserved practice:",inc18.mean(),"paired SE",inc18.std(ddof=1)/np.sqrt(len(inc18)))
with open(WORK/"notes.md","a") as f:
    f.write("\nPass 18 linear-target rank1 reserved practice: original .591111 -> edited .746667, "
            "gain .155556, paired SE .020028; 103 improved, 29 worsened. "
            f"Increment over best2 {inc18.mean():+.6f}, paired SE {inc18.std(ddof=1)/np.sqrt(len(inc18)):.6f}; "
            "candidates effectively tied, no further adjustment from test results.\n")
    f.write("\n## Before pass 19\nExact edit: k.weight = K0 + outer(u,v_linear), "
            "v_linear in candidate_linear_rank1/derivation.pt, linear target 4*p, ridge .001, strength .75. "
            "Other tensors original. Hidden VALIDATION both on all 300 cases; no records visible. "
            "Candidate fixed before validation; best2 remains fallback.\n")
```
