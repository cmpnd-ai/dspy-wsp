The new files align exactly with the original outputs. I’ll now solve the unchanged mixed-target correction on all 900 records and compute three held-out-fold estimates of that recipe. The fold estimates will measure generalization, not be used to tune it.
I’ll solve the unchanged mixed-target correction on all 900 records and compute three held-out-fold estimates. The fold estimates are a check on the fixed recipe, not a new hyperparameter search.
```python
# Hypothesis: the fixed mixed-target scalar repair generalizes across folds and becomes more stable with all900 practice records.
rows900 = rows + new15[0] + new16[0]
X900 = np.concatenate([X, new15[1]["head_inputs"], new16[1]["head_inputs"]]).astype(np.float32)
mask900 = np.concatenate([mask, new15[1]["option_mask"], new16[1]["option_mask"]]).astype(bool)
L900 = np.concatenate([L0, new15[1]["option_logits"], new16[1]["option_logits"]]).astype(np.float32)
for i, row in enumerate(rows900):
    if row["type"] == "noul":
        jfalse = row["options"].index("false")
        X900[i, jfalse] = 0
        L900[i, jfalse] = 0
X900 *= mask900[..., None]
L900 = np.where(mask900, L900, 0)
G900 = np.zeros(mask900.shape, dtype=np.float32)
for i, row in enumerate(rows900):
    gp = by_id[row["id"]]["labels"]["gold"][row["question"]]["probabilities"]
    G900[i, :len(row["options"])] = [gp[o] for o in row["options"]]
n900 = mask900.sum(1)
def center900(a):
    if a.ndim == 3:
        return (a - (a*mask900[..., None]).sum(1,keepdims=True)/n900[:,None,None])*mask900[...,None]
    return (a-(a*mask900).sum(1,keepdims=True)/n900[:,None])*mask900
Xc900 = center900(X900).astype(np.float32)
H900 = np.zeros_like(G900)
H900[np.arange(len(rows900)), G900.argmax(1)] = 1
Tm900 = center900(3*(0.5*G900+0.5*H900)).astype(np.float32)
res900 = (Tm900-center900(L900)).astype(np.float32)
rw900 = (1/np.sqrt(n900)).astype(np.float32)
ids900 = np.array([r["id"] for r in rows900])
unique900 = list(dict.fromkeys(ids900.tolist()))
def solve900(selection, lam):
    dd = torch.from_numpy((Xc900[selection]*rw900[selection,None,None])[mask900[selection]]).double()
    yy = torch.from_numpy((res900[selection]*rw900[selection,None])[mask900[selection]]).double()
    gg = dd.T @ dd
    rr = dd.T @ yy
    gg.diagonal().add_(lam)
    return torch.linalg.solve(gg, rr).float().numpy()
def evaluate900(logits, selection=None):
    if selection is None:
        selection = np.ones(len(rows900),bool)
    ll = np.where(mask900,logits,-1e20)
    correct = (ll.argmax(1)==G900.argmax(1)).astype(np.float64)
    original = (np.where(mask900,L900,-1e20).argmax(1)==G900.argmax(1)).astype(np.float64)
    ordered = list(dict.fromkeys(ids900[selection].tolist()))
    scores = np.array([correct[(ids900==i)&selection].mean() for i in ordered])
    deltas = np.array([(correct-original)[(ids900==i)&selection].mean() for i in ordered])
    prob = np.exp(ll-ll.max(1,keepdims=True)); prob /= prob.sum(1,keepdims=True)
    return {"accuracy": float(scores.mean()), "gain": float(deltas.mean()),
            "paired_se": float(deltas.std(ddof=1)/np.sqrt(len(deltas))),
            "brier_sum": float(((prob-G900)**2).sum(1)[selection].mean()), "items":len(scores)}
old600 = np.arange(len(rows900)) < len(rows)
d600_check = solve900(old600, .004)
print("Exact recipe reconstruction max difference:", float(np.max(np.abs(d600_check-d_mix600))))
assert np.max(np.abs(d600_check-d_mix600)) < 0.003
d_mix900 = solve900(np.ones(len(rows900),bool), .006)
torch.save(torch.from_numpy(d_mix900), work/"candidate_weights"/"mix900_direction.pt")
wmix900 = torch.from_numpy(w0+0.75*d_mix900).reshape(1,-1)
torch.save(wmix900, work/"candidate_weights"/"mix900_075_scalar.pt")
print("Full900 norm/cosine with600:", float(np.linalg.norm(d_mix900)),
      float(d_mix900@d_mix600/(np.linalg.norm(d_mix900)*np.linalg.norm(d_mix600))))
full900_logits = L900+0.75*np.einsum("qod,d->qo",X900,d_mix900)
print("Full900 fitted performance (NOT held out):", evaluate900(full900_logits))
foldmap = {}
rngcv = np.random.default_rng(20260310)
for wf in sorted({r["workflow"] for r in rows900}):
    wfids = [i for i in unique900 if by_id[i]["inputs"]["workflow"]==wf]
    rngcv.shuffle(wfids)
    for j, itemid in enumerate(wfids):
        foldmap[itemid] = j%3
folds900 = np.array([foldmap[i] for i in ids900])
cross_logits = np.zeros_like(L900)
cross_dirs = []
for fold in range(3):
    held = folds900==fold
    d_cv = solve900(~held, .004)
    cross_dirs.append(d_cv)
    cross_logits[held] = L900[held]+0.75*np.einsum("qod,d->qo",X900[held],d_cv)
    print("Fixed-recipe held fold", fold, evaluate900(cross_logits, held))
cv900_summary = evaluate900(cross_logits)
print("Combined cross-fit:", cv900_summary)
note("\nFixed full900 mixed-target recipe reproduced600 direction (numerical tolerance verified). lambda=.006, strength=.75. Full900 in-sample: "+str(evaluate900(full900_logits))+
     ". Fixed-recipe three-fold estimate (not an untouched recipe-selection set): "+str(cv900_summary)+". No new edits saved yet; best2 remains active.")
```
