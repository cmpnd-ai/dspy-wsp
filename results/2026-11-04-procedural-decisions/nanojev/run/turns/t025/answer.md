The larger-data edit improves validation to 68.67%, a gain of 23.56 points (paired SE 3.02). I’ll preserve it and test a diagnostic control: a correction based only on average question/option features. If that control fails to match the full edit, the gain cannot be explained solely by replacing answer priors.

```python
# Hypothesis: state-dependent feature variation, not just question/option offsets, accounts for most of the repair.
shutil.copytree(work/"edited",work/"best_900_ridge01",dirs_exist_ok=True)
# Each row of the least-squares design has a question and candidate-position identity.
design_groups=[]
for r in allrows:
    if r["type"]=="noul": design_groups.append((r["question"],"true"))
    else: design_groups += [(r["question"],j) for j in range(len(r["options"]))]
groups=collections.defaultdict(list)
for j,g in enumerate(design_groups):groups[g].append(j)
XP=np.zeros_like(XX); YP=np.zeros_like(YY)
for g,ind in groups.items():
    XP[ind]=XX[ind].mean(0); YP[ind]=YY[ind].mean()
delta_prior=np.linalg.solve(XP.T@XP+.1*np.eye(1024),XP.T@YP)
delta_evidence=np.linalg.solve((XX-XP).T@(XX-XP)+.1*np.eye(1024),(XX-XP).T@(YY-YP))
for name,d in [("full",delta900),("prior-only",delta_prior),("evidence-only",delta_evidence)]:
    vals,qs=eval_all(d)
    print(name,"norm",round(np.linalg.norm(d),3),"all-practice",round(np.mean(list(vals.values())),4),qs)
save_edit("scalar.weight",torch.from_numpy((W0+delta_prior).astype(np.float32)[None]))
with open(notes,"a") as f:
    f.write("\nPass 9 VALIDATION delta900 ridge.1: .45111 -> .68667, gain .23556 ± .03020;148 improved,51 worsened. Saved best_900_ridge01."
            "\nPass 10 control: scalar original+prior-only direction, fit using means within question/option identity, no state-varying evidence; same 450 practice diagnostic cohort (now part of larger fit).\n")
```
