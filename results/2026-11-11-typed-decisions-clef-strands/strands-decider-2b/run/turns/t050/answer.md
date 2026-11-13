I’ll compare the predeclared target transforms using the existing fit-only ridge system. These are independent closed-form solutions, not iterative training. I’ll check both development accuracy and the paired gain over the validated edit before advancing a candidate.



```python
# Hypothesis: smoother target contrasts improve the same rank-one state-evidence path without introducing higher-rank overfitting.
smooth_specs = {
    "log_p+.05": torch.log(fit["gold"]+.05),
    "log_p+.1": torch.log(fit["gold"]+.1),
    "log_p+.2": torch.log(fit["gold"]+.2),
    "4*p": 4*fit["gold"]
}
smooth_candidates = {}
smooth_results = []
def item_delta_stats(dd, logits_new, logits_old):
    target = dd["gold"].argmax(-1)
    newok = (logits_new.masked_fill(~dd["mask"],-torch.inf).argmax(-1)==target).float()
    oldok = (logits_old.masked_fill(~dd["mask"],-torch.inf).argmax(-1)==target).float()
    by_id = collections.defaultdict(list)
    for eid,xv in zip(dd["ids"],(newok-oldok).tolist()):
        by_id[eid].append(xv)
    arr=np.array([np.mean(vv) for vv in by_id.values()])
    return float(arr.mean()), float(arr.std(ddof=1)/np.sqrt(len(arr)))
for label, target_values in smooth_specs.items():
    residual3 = center_options(target_values-fit["L"]/fit["temp"][:,None],fit["mask"])
    smooth_rhs = X.double().T @ (residual3[fit["mask"]]*weights).double()
    smooth_proj = evecs.T @ smooth_rhs
    for lam in (.0003,.001):
        vec = (evecs @ (smooth_proj/(evals+lam))).float()
        for strength in (.75,1.):
            vcan = strength*vec
            ccan = correction(dev,vcan)
            stats = eval_cached(dev,dev["L"]+ccan)
            inc = item_delta_stats(dev,dev["L"]+ccan,dev["L"]+Cdev_ref)
            key=(label,lam,strength)
            smooth_candidates[key]=vcan
            smooth_results.append((key,stats,inc))
            print(key,"accuracy",round(stats["accuracy"],6),"inc_best2",tuple(round(z,6) for z in inc),
                  "Brier",round(stats["brier"],5))
smooth_results.sort(key=lambda z:(z[1]["accuracy"],-z[1]["brier"]),reverse=True)
smooth_best_key, smooth_best_stats, smooth_best_inc = smooth_results[0]
print("Best independent smooth candidate:",smooth_best_key,smooth_best_stats,"increment",smooth_best_inc)
with open(WORK/"notes.md","a") as f:
    f.write("\nSmooth-target closed-form development comparisons (no saved weight changes):\n")
    for spec,stats,inc in smooth_results:
        f.write(f"- {spec}: dev={stats['accuracy']:.6f}, inc_best2={inc[0]:+.6f} SE={inc[1]:.6f}, Brier={stats['brier']:.6f}\n")
```
