The all-practice refit validates at 65.60%, a gain of 8.07 ± 1.26 points, and is now the best confirmed candidate. I’ll preserve it and test whether its benefit comes mainly from fixed question-option biases or from contextual variation within each question-option group.

```python
# Hypothesis: the readout correction combines option-prior repair with context-dependent discrimination.
if (work/"best_all").exists():shutil.rmtree(work/"best_all")
shutil.copytree(work/"edited",work/"best_all")
groups=collections.defaultdict(list)
for i,r in enumerate(rows):
    for j in np.flatnonzero(M[i]):groups[(r[1],r[2],j)].append((i,j))
means=np.stack([np.mean([X[i,j] for i,j in inds],axis=0) for inds in groups.values()])
u,s,vt=np.linalg.svd(means,full_matrices=False)
basis=vt[s>1e-5]
prior_delta=basis.T@(basis@delta_all)
context_delta=delta_all-prior_delta
print("Prior/context norms",np.linalg.norm(prior_delta),np.linalg.norm(context_delta))
print("Prior-only predicted",report(prior_delta,~fit),"context-only",report(context_delta,~fit))
save_edit("head.scorer.3.weight",torch.tensor(w0+prior_delta).reshape(1,-1))
with notes.open("a") as f:
    f.write("\nPass15 all900 refit VALIDATION .57533→.65600, +.08067±.01260; 142 improved/60 worsened. Saved best_all. Diagnostic ablation: project all900 delta onto span of workflow/question/option mean centered features (prior component); complementary context component. Next pass both on 450 reserved practice (now used in all900 fit; diagnostic only), prior-only edit.\n")
```
