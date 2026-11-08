The activation files have the expected shapes. I’ll read their descriptions and compare correct-option versus chosen-option contrasts across the two halves of the read sample; reproducible contrasts are better candidates than features found in only a few failures.
```python
# Hypothesis: a repeatable contrast in final-scorer features can identify a specific underweighted direction without fitting a predictor.
print("Metadata descriptions:", json.dumps({k:v for k,v in meta2.items() if k!="rows"},ensure_ascii=False,indent=2))
requests2 = json.loads((F2/"requests.json").read_text())
print("Requests container:",type(requests2).__name__,"length",len(requests2))
print("First request snippet:",json.dumps(requests2[0],ensure_ascii=False)[:700] if isinstance(requests2,list) else str(list(requests2))[:500])
valid2 = np.asarray(arr2["option_mask"])
H2 = np.asarray(arr2["head_inputs"])
T2 = np.asarray(arr2["trunk_markers"])
print("Head features: norm mean/std",np.linalg.norm(H2[valid2],axis=1).mean(),np.linalg.norm(H2[valid2],axis=1).std(),
      "range",H2[valid2].min(),H2[valid2].max(),"negative fraction",(H2[valid2]<0).mean())
print("Trunk features: norm mean/std",np.linalg.norm(T2[valid2],axis=1).mean(),np.linalg.norm(T2[valid2],axis=1).std())
rowinfo2 = []
for i, m in enumerate(meta2["rows"]):
    eid = read_ids_a[m["request"]]
    gold = by_id[eid]["labels"]["gold"][m["question"]]["probabilities"]
    g = np.array([gold[o] for o in m["options"]])
    p = np.asarray(arr2["probabilities"][i,:len(g)])
    rowinfo2.append(dict(i=i, id=eid, wf=by_id[eid]["inputs"]["workflow"], q=m["question"], typ=m["type"],
                         opts=m["options"], g=g, gi=int(g.argmax()), pi=int(p.argmax()), ok=g.argmax()==p.argmax()))
print("Read-vs-decide argmax agreement:",np.mean([r["opts"][r["pi"]]==top(answers(base_by_id[r["id"]])[r["q"]]) for r in rowinfo2]))
```
