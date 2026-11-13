I’ll derive a rank-one correction to `k.weight` by a single closed-form ridge solve. It will read option contrasts that the original head misses, targeting the gold relative logits rather than globally sharpening probabilities. Development results will determine whether that direction generalizes.

```python
# Hypothesis: a rank-one key correction can recover state-conditioned option evidence without changing the backbone or global temperature.
dev = load_read(5,splits["dev"])
dev["Q"] = head_q(dev["d"])
dev["g"] = dev["Q"]@u/math.sqrt(768)
def accuracy_stats(data,L):
    pred=L.masked_fill(~data["mask"],-1e9).argmax(1)
    target=data["gold"].argmax(1)
    base=data["L"].masked_fill(~data["mask"],-1e9).argmax(1)
    ok=(pred==target).float()
    dif=ok-(base==target).float()
    itemdif=dif.reshape(-1,5).mean(1)
    probs=torch.softmax((L/data["temp"][:,None]).masked_fill(~data["mask"],-1e9),1)
    brier=((probs-data["gold"])**2).sum(1).mean().item()
    return (ok.mean().item(),itemdif.mean().item(),itemdif.std(unbiased=True).item()/math.sqrt(len(itemdif)),brier)
def x_for(data):
    return center_options(data["o"],data["mask"])*data["g"][:,None,None]/data["temp"][:,None,None]
Xfit3 = x_for(fit)
Xdev3 = x_for(dev)
weights = (1/fit["mask"].sum(1).float()).sqrt()[:,None].expand_as(fit["mask"])[fit["mask"]]
X = Xfit3[fit["mask"]]*weights[:,None]
target_log = center_options(torch.log(fit["gold"].clamp_min(.01)),fit["mask"])
y3 = target_log-center_options(fit["L"]/fit["temp"][:,None],fit["mask"])
y = y3[fit["mask"]]*weights
start=time.time()
gram = X.T.double()@X.double()
rhs = X.T.double()@y.double()
evals,evecs=torch.linalg.eigh(gram)
print("Closed-form eigensolve seconds:",round(time.time()-start,2))
print("Gram eigenvalues quantiles",torch.quantile(evals,torch.tensor([0,.25,.5,.75,.9,.99,1.],dtype=torch.float64)).tolist())
projrhs=evecs.T@rhs
ridge_vs={}
results_ridge=[]
for lam in [.001,.01,.1,1.,10.,100.]:
    v=(evecs@(projrhs/(evals+lam))).float()
    ridge_vs[lam]=v
    for strength in [.5,1.0,1.5]:
        LF=fit["L"]+strength*(Xfit3@v)*fit["temp"][:,None]
        LD=dev["L"]+strength*(Xdev3@v)*dev["temp"][:,None]
        sf=accuracy_stats(fit,LF)
        sd=accuracy_stats(dev,LD)
        results_ridge.append((lam,strength,sf,sd))
        print(f"ridge={lam:7g} strength={strength:.1f} | fit={sf[0]:.3f} dev={sd[0]:.3f} "
              f"gain={sd[1]:+.3f} SE={sd[2]:.3f} Brier={sd[3]:.3f} | deltaK norm={v.norm().item()*strength:.2f}")
print("Original fit/dev:",accuracy_stats(fit,fit["L"]),accuracy_stats(dev,dev["L"]))
lognote("Closed-form experiment: X = within-question centered normalized option activations * "
        "(query dot mean-query unit direction)/(sqrt(768)*temperature). Targets are centered "
        "log(gold probability, floor .01) minus original relative logits. One eigendecomposition "
        "of X'X and closed-form ridge solutions, no gradients or iterative fitting. Fit=540 only.")
lognote("Rank-one key development grid:\n"+"\n".join(
    f"- lambda={lam}, strength={s}: fit={sf[0]:.4f}, dev={sd[0]:.4f}, paired gain={sd[1]:+.4f} SE={sd[2]:.4f}, Brier={sd[3]:.4f}"
    for lam,s,sf,sd in results_ridge))
```
