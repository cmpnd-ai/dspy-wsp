C also improves the fitting partition substantially. Next I’ll test whether its correction is concentrated in a small set of scorer neurons. I’ll rank coordinates by their effect on centered option logits and ablate the repair to the strongest coordinates.

```python
# Hypothesis: a subset of final scorer neurons carries most of the useful correction.
with (work/"notes.md").open("a") as f:
    f.write("\nPass 17 full C fit partition .51032 -> .80238, gain .29206 SE .02131 (fit-included).\n")
dc=np.asarray(wC)-np.asarray(w0)
energy=np.mean(np.asarray(Xall)**2,axis=0)*dc**2
rank=np.argsort(-energy)
for n in [64,256,512,1024]:
    ws=np.asarray(w0).copy()
    ws[rank[:n]]+=dc[rank[:n]]
    vals=measure(np.einsum("qod,d->qo",At,ws),rt,Mt,yt)
    print(n, str(vals)[:220])
print("top neuron coordinates",rank[:20].tolist())
print("delta norm",np.linalg.norm(dc),"original norm",np.linalg.norm(w0),
      "cosine",np.dot(wC,w0)/(np.linalg.norm(wC)*np.linalg.norm(w0)))
```
