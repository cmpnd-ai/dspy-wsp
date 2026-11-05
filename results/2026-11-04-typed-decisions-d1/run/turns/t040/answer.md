The context-only component contributes +4.13 ± 0.83 points, supporting a contextual readout repair rather than just new option priors. I’ll restore the best full edit and examine whether a smaller subset of scorer neurons can reproduce its effect.

```python
# Hypothesis: a subset of high-contribution scorer neurons may account for most of the useful correction.
impact=np.abs(delta_all)*np.sqrt(np.mean(X[M]**2,axis=0))
rank=np.argsort(impact)[::-1]
sparse={}
for k in [64,128,256,512,768]:
    dd=np.zeros(1024); dd[rank[:k]]=delta_all[rank[:k]]
    sparse[k]=dd
    print("top",k,"fit",report(dd,fit),"reserved",report(dd,~fit))
print("all-data doses",[(a,report(a*delta_all,~fit)) for a in [.5,1,1.5]])
save_edit("head.scorer.3.weight",torch.tensor(w0+.5*delta_all).reshape(1,-1))
with notes.open("a") as f:
    f.write("\nPass17 context-only ablation gain .04133±.00827; supports contextual discrimination beyond prior correction. Current diagnostic half-strength all900 edit. Next both same450. Plan also 1.5 strength, sparse neuron ablations; do not replace best without validation.\n")
```
