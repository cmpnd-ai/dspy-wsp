The scorer reconstruction matches within 0.000003 logits. The diagnostic calculation exposed two bookkeeping issues: padded features contain NaNs, and read requests follow the baseline record order rather than the requested ID order. I’ll fix both before deriving any edit.

```python
# Hypothesis: with correct row alignment and padding removal, a readout-only correction can recover suppressed choices.
fit_order=[r["id"] for r in base if r["id"] in set(fitids)]
A=np.nan_to_num(A)
Lclean=np.nan_to_num(L)
for i,r in enumerate(rows):
    gold=byid[fit_order[r["request"]]]["labels"]["gold"][r["question"]]["probabilities"]
    y[i]=0
    y[i,:len(r["options"])]=[gold[o] for o in r["options"]]
Ac=A-(A*M[:,:,None]).sum(1)[:,None,:]/counts[:,None,None]
yc=y-y.sum(1)[:,None]/counts[:,None]
Lc=Lclean-(Lclean.sum(1)/counts)[:,None]
X=Ac[mask].astype(np.float64)
target=(yc-Lc)[mask].astype(np.float64)
gram=X.T@X
evals,evecs=np.linalg.eigh(gram)
rhs=evecs.T@(X.T@target)
deltas={}
for reg in [1.,10.,100.,1000.]:
    deltas[reg]=(evecs@(rhs/(evals+reg))).astype(np.float32)
print("baseline",summary(Lclean),"w norm",np.linalg.norm(w0))
for reg,d in deltas.items(): print(reg,"delta norm",round(float(np.linalg.norm(d)),3),summary(Lclean+A@d))
with open(work/"notes.md","a") as f:
    f.write("Read pass 2: reconstructed scorer logits max error 2.86e-6. Fit order is original baseline order; padded NaNs removed. Testing analytic residual ridge correction of final scorer readout; no edits yet.\nNext pass: read 420 untouched practice test cases, original weights.\n")
json.dump({"type":"read","model":"original","from":{"pass":1,"ids":testids}},open(work/"read_test.json","w"))
```
