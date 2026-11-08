The read tool returned requests in recorded order, not the order of my ID list, so the initial ID-to-activation mapping was wrong. I’ll correct it by matching the full saved requests before using any labels. The exported features are before the scorer’s LayerNorm, which suggests testing a directional edit to the scorer’s first linear layer.
```python
# Hypothesis: after exact request matching, reproducible correct-minus-chosen feature contrasts will localize an underweighted scorer direction.
def canonical(x):
    return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":"))
request_to_id = {}
for eid in discovery_ids:
    for call in base_by_id[eid]["calls"]:
        request_to_id[canonical(json.loads(call["prompt"]))] = eid
request_ids2 = [request_to_id[canonical(req)] for req in requests2]
assert set(request_ids2)==set(read_ids_a)
rowinfo2 = []
for i, m in enumerate(meta2["rows"]):
    eid = request_ids2[m["request"]]
    gold = by_id[eid]["labels"]["gold"][m["question"]]["probabilities"]
    g = np.array([gold[o] for o in m["options"]])
    p = np.asarray(arr2["probabilities"][i,:len(g)])
    rowinfo2.append(dict(i=i,id=eid,wf=by_id[eid]["inputs"]["workflow"],q=m["question"],typ=m["type"],
                        opts=m["options"],g=g,gi=int(g.argmax()),pi=int(p.argmax()),ok=g.argmax()==p.argmax()))
print("Corrected read-vs-decide argmax agreement:",np.mean([r["opts"][r["pi"]]==top(answers(base_by_id[r["id"]])[r["q"]]) for r in rowinfo2]))
# Geometric analysis only: remove per-vector offset and magnitude, which are nuisance directions before LayerNorm.
C2 = H2 - np.nanmean(H2,axis=-1,keepdims=True)
U2 = C2 / np.linalg.norm(C2,axis=-1,keepdims=True)
contrast_groups = {}
for key in sorted(groups):
    rr = [r for r in rowinfo2 if (r["wf"],r["q"])==key and not r["ok"]]
    d = np.stack([U2[r["i"],r["gi"]]-U2[r["i"],r["pi"]] for r in rr])
    a,b = d[::2].mean(0),d[1::2].mean(0)
    c = d.mean(0)
    contrast_groups[key] = c
    print(key,"errors",len(rr),"contrast norm",round(float(np.linalg.norm(c)),4),
          "half cosine",round(float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),3))
W1 = load_original("head.scorer.1.weight").numpy()
w3 = load_original("head.scorer.3.weight").numpy().reshape(-1)
gain0 = load_original("head.scorer.0.weight").numpy()
effective = gain0 * (W1.T @ w3)
print("Static scorer readout norm",np.linalg.norm(effective))
all_errors2 = [r for r in rowinfo2 if not r["ok"]]
D2 = np.stack([U2[r["i"],r["gi"]]-U2[r["i"],r["pi"]] for r in all_errors2])
mean_diff2 = D2.mean(0)
print("Pooled error contrast norm",np.linalg.norm(mean_diff2),
      "cosine static readout",float(mean_diff2@effective/(np.linalg.norm(mean_diff2)*np.linalg.norm(effective))))
note("Read pass 2 ordering corrected by matching complete requests; corrected mappings are request_ids2. head_inputs are pre-scorer-LayerNorm, NOT post-GELU. No network computations simulated. Contrasts use centered unit-length exported vectors as geometric analysis.")
```
