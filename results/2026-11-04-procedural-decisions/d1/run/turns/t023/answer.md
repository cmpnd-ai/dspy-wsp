Candidate B is confirmed on validation: 69.0% versus 54.3%, a 14.7-point gain with a 2.8-point paired SE. I’ll preserve it and examine whether the repair’s record-sensitive directions—not just changed option biases—cause the improvement.

```python
# Hypothesis: the repair requires record-sensitive neuron combinations, not merely fixed option preference changes.
shutil.copytree(work/"edited",work/"best_B",dirs_exist_ok=True)
with open(work/"notes.md","a") as f:
    f.write("Pass 7 B validation .69000 vs .54333, gain .14667 SE .02833. Saved best_B. Next causal tests separate fixed-option feature subspace from record-sensitive complement.\n")
# Build a subspace spanning the mean centered feature vector for each question/option.
means=[]
for q in dict.fromkeys(r["question"] for r in rows):
    ix=np.array([r["question"]==q for r in rows])
    for o in range(int(counts[ix][0])):
        means.append(Ac[ix,o].mean(0))
U,S,Vt=np.linalg.svd(np.array(means),full_matrices=False)
rank=int((S>1e-4).sum())
V=Vt[:rank].T
db=direct[.1]-w0
bias_delta=V@(V.T@db)
evidence_delta=db-bias_delta
print("mean-option subspace rank",rank,"delta norms",np.linalg.norm(db),np.linalg.norm(bias_delta),np.linalg.norm(evidence_delta))
for name,dd in [("bias only",bias_delta),("evidence only",evidence_delta),("both",db)]:
    st,qt=measure(Lt+At@dd,rt,Mt,yt)
    print(name,st.mean(),qt)
reserveids=[x["id"] for x in data if x["id"] not in set(fitids+testids)]
json.dump({"type":"read","model":"original","from":{"pass":1,"ids":reserveids}},open(work/"read_reserve.json","w"))
with open(work/"notes.md","a") as f: f.write("Next pass original read of 60 reserve cases; current edit B (read original ignores edit).\n")
```
