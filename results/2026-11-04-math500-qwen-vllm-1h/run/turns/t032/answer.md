The teacher-forced pass gives token-level MLP activations for both transcripts. I’ll align those activations with the text and compare repeated “Wait”/self-correction passages against the earlier reasoning, looking for a localized layer-and-channel pattern to test. This remains correlational; I’ll require a paired generation test before accepting any edit. The prior block-23 ablation remains a failed probe. 

```python
# Align teacher-forced activations with transcript tokens and compare self-correction markers against earlier reasoning by layer.
import numpy as np, json
from pathlib import Path
base=Path("/lambda/nfs/dspy-wsp-is/runs/hour/work")
acts=np.load(base/"fwd_5/mlp_act.npy")
ids=np.load(base/"fwd_5/token_ids.npy")
lengths=np.load(base/"fwd_5/lengths.npy")
meta=json.load(open(base/"fwd_5/meta.json"))
print("shapes",acts.shape,ids.shape,"lengths",lengths.tolist(),"meta",meta)
for i,r in enumerate(rs[:2]):
    text=r["calls"][0]["prompt"]+r["calls"][0]["completion"][:4500]
    # Locate occurrences of verbal hesitation in raw transcript; report average MLP activity near vs elsewhere.
    chars=["Wait","Let's verify","No.","double-check","one more time"]
    print("\nitem",i)
    for phrase in chars:
        pos=[]; start=0
        while (j:=text.find(phrase,start))>=0:
            pos.append(j); start=j+len(phrase)
        # Map character offsets to token offsets using tokenizer offset mapping if available, otherwise approximate prefix token counts.
        spans=[]
        for j in pos:
            spans.append(len(tok.encode(text[:j],add_special_tokens=False)))
        spans=[x for x in spans if x < int(lengths[i])]
        if spans:
            win=np.concatenate([np.arange(max(0,x-3),min(int(lengths[i]),x+12)) for x in spans])
            baseidx=np.setdiff1d(np.arange(int(lengths[i])),win)
            print(phrase,"occ",len(spans),"layer abs near/base",
                  np.round(np.mean(np.abs(acts[i,:,win,:]),axis=(1,2)),4).tolist(),
                  np.round(np.mean(np.abs(acts[i,:,baseidx,:]),axis=(1,2)),4).tolist())
```