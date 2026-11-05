I’ll measure the reversed edit’s cue responses, then install the half-strength correction. Comparing negative, half, full, and overshoot strengths will test a dose-response rather than just a right/wrong correlation.
```python
# Hypothesis: cue-induced target-margin changes vary predictably with correction strength.
def cue_stats(passno):
    logits=np.load(p/f"fwd_{passno}/option_logits.npy")
    meta=json.load((p/f"fwd_{passno}/meta.json").open())
    rows=meta["rows"] if isinstance(meta,dict) and "rows" in meta else meta
    if isinstance(rows,dict):
        print("meta keys",rows.keys()); return
    out=collections.defaultdict(list)
    for pair in probe_pairs:
        i,j=pair["rows"]; ta,tb=pair["targets"]
        oi=rows[i]["options"]; oj=rows[j]["options"]
        ma=logits[i,oi.index(ta)]-logits[i,oi.index(tb)]
        mb=logits[j,oj.index(tb)]-logits[j,oj.index(ta)]
        ca=oi[np.argmax(logits[i,:len(oi)])]==ta
        cb=oj[np.argmax(logits[j,:len(oj)])]==tb
        out[pair["question"]].append([(ma+mb)/2,ca and cb,(int(ca)+int(cb))/2])
    return {q:np.mean(v,axis=0).round(3).tolist() for q,v in out.items()}
print("Reversal",cue_stats(19))
save_edit("scalar.weight",torch.tensor(W0.reshape(-1)+.5*delta900,dtype=torch.float32).reshape(1,-1))
with notes.open("a") as f:
    f.write(f"Pass19 reversal probes {cue_stats(19)}.\nPass20 scalar original+.5*delta900, same420 matched probe requests.\n")
```
