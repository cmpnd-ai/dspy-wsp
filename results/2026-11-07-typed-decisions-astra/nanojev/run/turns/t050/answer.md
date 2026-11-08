The cross-fit strength response is broad rather than sharply peaked. I’ll separate the correction into an average question/option component and a component whose average is zero within each question/option. Varying them separately tests whether more state-dependent sensitivity helps, or merely amplifies noise.
```python
# Hypothesis: gains require state-dependent contrasts in addition to removing fixed option preferences; their strengths may differ.
template_keys900 = sorted(set((r["workflow"],r["question"]) for r in rows900))
def template_projection900(direction, selection):
    means = []
    for wf,qid in template_keys900:
        ix = selection&(wf900==wf)&(qid900==qid)
        nn = int(n900[np.flatnonzero(ix)[0]])
        means.extend(Xc900[ix,:nn].mean(0))
    mm = torch.from_numpy(np.array(means)).double()
    _,ss,vv = torch.linalg.svd(mm,full_matrices=False)
    rank = int((ss>ss[0]*1e-5).sum())
    qq = vv[:rank].T
    dt = torch.from_numpy(direction).double()
    prior = (qq@(qq.T@dt)).float().numpy()
    return prior, direction-prior, rank
cv_prior_corr = np.zeros_like(L900)
cv_evidence_corr = np.zeros_like(L900)
cv_components = []
for fold,direction in enumerate(cross_dirs):
    held = folds900==fold
    dp,de,rank = template_projection900(direction,~held)
    cv_components.append((dp,de))
    cv_prior_corr[held] = np.einsum("qod,d->qo",X900[held],dp)
    cv_evidence_corr[held] = np.einsum("qod,d->qo",X900[held],de)
    print("Fold",fold,"template rank",rank,"norm prior/evidence",round(float(np.linalg.norm(dp)),3),round(float(np.linalg.norm(de)),3))
assert np.max(np.abs(cv_prior_corr+cv_evidence_corr-cross_correction)) < 0.001
component_results = []
for ap in [.5,.75,.875,1,1.25]:
    for ae in [.5,.75,.875,1,1.25]:
        result = evaluate900(L900+ap*cv_prior_corr+ae*cv_evidence_corr)
        component_results.append({"prior_strength":ap,"evidence_strength":ae,**result})
print("Top component-strength combinations:")
for result in sorted(component_results,key=lambda r:r["accuracy"],reverse=True)[:10]:
    print({k:round(v,5) if isinstance(v,float) else v for k,v in result.items()})
for ap,ae in [(0,0),(.875,0),(0,.875),(.875,.875)]:
    print("Ablation strengths",ap,ae,evaluate900(L900+ap*cv_prior_corr+ae*cv_evidence_corr))
dp900,de900,rank900 = template_projection900(d_mix900,np.ones(len(rows900),bool))
torch.save({"prior":torch.from_numpy(dp900),"evidence":torch.from_numpy(de900)},work/"candidate_weights"/"mix900_components.pt")
note("\nCross-fit decomposition diagnostic, finite predeclared strength grid: "+str(component_results)+
     ". Components are orthogonal projection onto training-fold mean centered question/option span and its complement; complement is not claimed to encode a single semantic cue. best3 unchanged.")
```
