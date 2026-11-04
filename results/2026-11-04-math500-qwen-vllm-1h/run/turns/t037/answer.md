The channel-level analysis failed because I indexed the activation array with an extra dimension. I’ll correct that and check whether any MLP channels are consistently more active near “Wait” in both transcripts. This is still exploratory; I’ll only test a narrow edit if the pattern is repeatable. The block-23 probe remains a failed intervention. 

```python
# Correct the channel-level indexing and find MLP channels enhanced near repeated "Wait" tokens in both transcripts.
import numpy as np, json
base="/lambda/nfs/dspy-wsp-is/runs/hour/work"
acts=np.load(base+"/fwd_5/mlp_act.npy")
lens=np.load(base+"/fwd_5/lengths.npy")
records=[json.loads(x) for x in open(base+"/gen_1.jsonl")]
scores=[]
for i,r in enumerate(records[:2]):
    text=r["calls"][0]["prompt"]+r["calls"][0]["completion"][:4500]
    offsets=tok(text,add_special_tokens=False,return_offsets_mapping=True)["offset_mapping"]
    starts=[]; pos=0
    while True:
        j=text.find("Wait",pos)
        if j<0: break
        starts.append(j); pos=j+4
    ids=[]
    for j in starts:
        ts=[k for k,(a,b) in enumerate(offsets) if b>j and a<j+4]
        if ts: ids.extend(range(max(0,min(ts)-2),min(int(lens[i]),max(ts)+3)))
    near=np.unique(ids)
    far=np.setdiff1d(np.arange(int(lens[i])),near)
    score=np.mean(np.abs(acts[i][:,near,:]),axis=1)-np.mean(np.abs(acts[i][:,far,:]),axis=1)
    scores.append(score)
for layer in [0,1,2,3,4,18,23]:
    tops=[set(np.argsort(s[layer])[-60:].tolist()) for s in scores]
    common=tops[0]&tops[1]
    print("layer",layer,"shared among top60",len(common),"channels",sorted(common)[:30])
```