Pass 20 tested the moderate, question-gated security repair. I’ll calculate its paired gain on the 450 confirmation cases and compare it with the previously validated payment/violation repair. I won’t use individual confirmation records to design edits.
```python
# Hypothesis: the gated security extension adds a reproducible gain beyond the base repair on untouched confirmation cases.
import numpy as np
def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
confirm20_original = read_jsonl(work / "gen_20_original.jsonl")
confirm20_edited = read_jsonl(work / "gen_20_edited.jsonl")
confirm12_edited = read_jsonl(work / "gen_12_edited.jsonl")
print("RECORD SCHEMA", {k:type(v).__name__ for k,v in confirm20_original[0].items()})
def paired_score_summary(before, after):
    ib = {str(r["id"]):r for r in before}
    ia = {str(r["id"]):r for r in after}
    ids = sorted(set(ib) & set(ia))
    b = np.array([float(ib[i]["score"]) for i in ids])
    a = np.array([float(ia[i]["score"]) for i in ids])
    d = a-b
    return dict(n=len(d), before=float(b.mean()), after=float(a.mean()),
                gain=float(d.mean()), paired_se=float(d.std(ddof=1)/np.sqrt(len(d))),
                improved=int((d>1e-8).sum()), worsened=int((d<-1e-8).sum()))
summary20 = paired_score_summary(confirm20_original, confirm20_edited)
increment20 = paired_score_summary(confirm12_edited, confirm20_edited)
print("PASS20 vs ORIGINAL", summary20)
print("PASS20 vs BASE REPAIR", increment20)
with open(work / "notes.md", "a") as f:
    f.write("\nPass20 confirmation summary: "+json.dumps(summary20)
            +"\nIncrement versus validated base repair (same cases): "+json.dumps(increment20)
            +"\nOnly aggregate scores examined; no confirmation examples used to construct or tune edits.\n")
```
