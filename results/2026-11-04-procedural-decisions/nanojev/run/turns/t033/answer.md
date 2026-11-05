The validated edit scores 80.30% on its fitting data. To localize what it uses, I’ll decompose the scalar correction into high- and low-variance EOS feature directions. A useful repair may depend on weak factual signals rather than the dominant directions that carry question and candidate identity.

```python
# Hypothesis: useful factual evidence occupies low-variance directions that the original scalar head underweights.
ee,UU=np.linalg.eigh(XX.T@XX)
coeff=UU.T@delta900
print("Eigenvalue quantiles",np.quantile(ee,[0,.25,.5,.75,.9,.99,1]))
spectral={}
for cutoff in [.001,.01,.1,1,10]:
    dlow=UU[:,ee<cutoff]@coeff[ee<cutoff]
    dhigh=delta900-dlow
    spectral[cutoff]=(dlow,dhigh)
    print("cutoff",cutoff,"low dims",sum(ee<cutoff),"low norm",round(np.linalg.norm(dlow),3),
          "high-only score",round(np.mean(list(eval_all(dhigh)[0].values())),4))
# Diagnostic high-variance-only readout, excluding all directions with eigenvalue below 1.
delta_high=spectral[1][1]
save_edit("scalar.weight",torch.from_numpy((W0+delta_high).astype(np.float32)[None]))
base_scores={r["id"]:r["score"] for r in records}
rr13=[json.loads(s) for s in open(work/"gen_13.jsonl")]
diff13=np.array([r["score"]-base_scores[r["id"]] for r in rr13])
print("In-sample paired gain",diff13.mean(),"SE",diff13.std(ddof=1)/30)
with open(notes,"a") as f:
    f.write(f"\nPass 13 all-practice (in-sample): .48000 -> .80296, gain {diff13.mean():.5f} ± {diff13.std(ddof=1)/30:.5f}."
            "\nPass 14 causal ablation: original scalar + high-variance portion of delta900 only (eigenvalue >=1 in practice contrast covariance). Same 450 diagnostic practice cohort; remove low-variance evidence component. Not a candidate to keep.\n")
```
