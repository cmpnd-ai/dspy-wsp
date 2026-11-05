The repair is nearly orthogonal to the original readout, rather than a rescaling of it. I’ll test 64-, 256-, and 512-coordinate ablations to see whether it depends on a distributed readout. These are diagnostic tests; C remains the confirmed candidate.

```python
# Hypothesis: distributed coordinate cancellation is essential, so sparse correction should underperform full C.
for n in [64,256,512,1024]:
    ws=np.asarray(w0).copy()
    ws[rank[:n]]+=dc[rank[:n]]
    scorevec=measure(np.einsum("qod,d->qo",At,ws),rt,Mt,yt)[0]
    print(n, float(np.mean(scorevec)))
sparse64=np.asarray(w0).copy()
sparse64[rank[:64]]+=dc[rank[:64]]
save_edit("head.scorer.3.weight",torch.tensor(sparse64).reshape(1,-1))
with (work/"notes.md").open("a") as f:
    f.write("\nNext pass 18: only top 64 scorer-coordinate corrections (ranked by centered feature energy times delta squared); paired 420 testids, fit-included. Diagnostic, not a replacement candidate.\n")
```
