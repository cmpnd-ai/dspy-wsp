Full strength increases the fit gain to 5.22 points (SE 0.73), consistent with the predicted response. I’ll inspect the question-level changes and test twice strength to see where the benefit starts to saturate.

```python
# Hypothesis: the correction repairs specific decision contrasts rather than uniformly sharpening outputs.
def getrecords(k,model):return [json.loads(s) for s in (work/f"gen_{k}_{model}.jsonl").read_text().splitlines()]
def qcorrect(r):
    return {q:ok=="right" for q,ok in re.findall(r"(\w+): (right|WRONG)",r["feedback"])}
for k in [5,6]:
    oo=getrecords(k,"original");ee=getrecords(k,"edited")
    dd=collections.defaultdict(list)
    for o,e in zip(oo,ee):
        a,b=qcorrect(o),qcorrect(e)
        for q in a:dd[(o["inputs"]["workflow"],q)].append(int(b[q])-int(a[q]))
    print("pass",k,[(key,round(np.mean(vv),3)) for key,vv in sorted(dd.items())])
save_edit(gn,g0*(1+torch.tensor(2*v,dtype=torch.float32)))
note("Pass 6 strength1 fit +.052232 SE .007295. Next pass7 strength2 same direction and same 448 fit cases. Inspect question-specific gains to verify non-global effect.")
```
