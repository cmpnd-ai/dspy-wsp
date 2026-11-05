I’ll summarize the controlled-pair results and decompose the scalar correction into covariance directions. This will show whether a few broad candidate-bias directions suffice or whether weaker, state-dependent directions are necessary.

```python
# Hypothesis: repaired cue sensitivity depends on low-variance representation directions, not just dominant candidate identity.
for passno in [11,12]:
    mm=json.load(open(work/f"fwd_{passno}/meta.json"))["rows"]
    pp=np.load(work/f"fwd_{passno}/probabilities.npy")
    print("ORIGINAL" if passno==11 else "REPAIRED")
    for i,r in enumerate(mm):
        if r["question"] in ["category","needs_review","risk"]:
            print(synlabels[r["request"]],r["question"],dict(zip(r["options"],np.round(pp[i,:len(r["options"])],3))))
# Covariance-direction ablation: each projection is algebraic, not refitted.
coef=Ua.T@dw600
groups_test=testdesign[2]+testdesign1[2]
for cutoff in [.01,.03,.1,.3,1,3,10]:
    dd=Ua[:,ea>=cutoff]@coef[ea>=cutoff]
    print("keep covariance eigenvalues >=",cutoff,"directions",sum(ea>=cutoff),
          "correction norm",round(np.linalg.norm(dd),2),"test",eval_dw(dd,groups_test))
with open(work/"notes.md","a") as f:
    f.write("\nPasses 11–12 completed controlled cue reads; next: algebraic covariance-direction ablations to characterize the validated scalar correction. Best remains original + dw600_linear.\n")
```
