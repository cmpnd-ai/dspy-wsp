The refinement improves reserved-test accuracy to 74.78%, a 15.67-point gain with paired SE 1.94 points. I’ll validate it before replacing the first confirmed edit.

```python
# Hypothesis: the plausible-option-focused rank-one repair improves generalization, not just development selection.
r9={r["id"]:r for r in [json.loads(l) for l in open(WORK/"gen_9_edited.jsonl")]}
r13={r["id"]:r for r in [json.loads(l) for l in open(WORK/"gen_13_edited.jsonl")]}
inc=np.array([r13[i]["score"]-r9[i]["score"] for i in r9])
print("Refinement vs best_1 on practice test: gain",inc.mean(),"SE",inc.std(ddof=1)/np.sqrt(len(inc)))
lognote("Pass 13 refined rank1 practice-test: .591111 -> .747778, gain .156667, paired SE .019381; "
        "99 improved, 25 worsened. Increment over best_1="+str(inc.mean())+
        ", paired SE="+str(inc.std(ddof=1)/np.sqrt(len(inc)))+".")
lognote("## Before pass 14\nExact edit: k.weight = K0 + outer(u,v_ref), floor .1, ridge .0003, strength .75. "
        "All other tensors original. Hidden VALIDATION both, all 300 cases. "
        "Candidate chosen on fit/dev, not validation. best_1 remains fallback.")
```
