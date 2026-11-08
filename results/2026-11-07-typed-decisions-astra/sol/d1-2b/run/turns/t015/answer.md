The read tool returned requests in baseline-file order, not the order of IDs submitted. That misaligned the labels; no fitted edit was retained. I’ll match requests to records by their serialized states and recompute the closed-form solutions.

```python
# Hypothesis: correctly aligned labels yield a useful causal gain correction.
statekey=lambda x:json.dumps(x,sort_keys=True)
state_ex={statekey(e["inputs"]["state"]):e for e in data}
aligned=[state_ex[statekey(r["state"]["inputs"]["state"])] for r in requests]
gold=np.zeros_like(L)
for i,row in enumerate(meta["rows"]):
    g=aligned[row["request"]]["labels"]["gold"][row["question"]]["probabilities"]
    for j,o in enumerate(row["options"]):gold[i,j]=g[o]
target=np.argmax(gold,1)
desired=temp*np.log(np.maximum(gold,.015))
desired-=(desired*mask).sum(1,keepdims=True)/mask.sum(1)[:,None]
Y=(desired-basecenter)[mask]
rhs=Xs.T@Y
solutions={}
print("aligned base accuracy",np.mean(np.nanargmax(L,1)==target))
for ridge in [0.1,1.,10.,100.]:
    v=np.zeros(2048);v[dims]=np.linalg.solve(gram+ridge*np.eye(len(dims)),rhs)
    solutions[ridge]=v
    for strength in [.5,1.,2.]:
        ll=logits_gain(v*strength)
        print("ridge,strength",ridge,strength,"accuracy",np.mean(np.nanargmax(ll,1)==target))
v=solutions[1.]
np.save(work/"gain_direction.npy",v)
save_edit(gn,g0*(1+torch.tensor(v*.5,dtype=torch.float32)))
note("Alignment fixed by exact state matching: READ from ids follows baseline order. Next pass 5: half-strength ridge=1 correction on 384 last-post-FFN gain dimensions; DECIDE both on 448 fit cases only. Direction stored gain_direction.npy.")
json.dump({"type":"decide","model":"both","problems":fitids},open(work/"fit_decide.json","w"))
print("largest gain changes",[(int(j),round(float(v[j]),3)) for j in np.argsort(abs(v))[-12:]])
```
