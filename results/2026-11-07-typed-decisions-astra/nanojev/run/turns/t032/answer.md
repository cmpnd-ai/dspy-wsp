The refitted candidate scores 56.9% on the independent practice test, a gain of 21.1 ± 1.4 percentage points over the original. I’ll validate it before deciding whether it should replace the current best checkpoint.
```python
# Hypothesis: the larger closed-form readout estimate generalizes at least as well as the first validated repair.
note("Forward12 mix600_075 independent practice300: original0.358000, edited0.568667, paired gain0.210667 SE0.014184;202 improved/29 worsened.")
r12={r["id"]:r for r in read_records(12,"edited")}
r7={r["id"]:r for r in read_records(7,"edited")}
dd=np.array([r12[i]["score"]-r7[i]["score"] for i in r12])
print("mix600 vs soft450 on practice test: gain",dd.mean(),"paired SE",dd.std(ddof=1)/np.sqrt(len(dd)))
note(f"mix600 vs prior soft-full on independent practice300: {dd.mean():.6f} SE{dd.std(ddof=1)/np.sqrt(len(dd)):.6f}; small incremental difference, not claimed significant.")
note("Planned forward13: paired hidden VALIDATION300. Exact current edit scalar.weight=w_original+.75*d_mix600, candidate_weights/mix600_075_scalar.pt. All others original. Best1 still preserved.")
```
