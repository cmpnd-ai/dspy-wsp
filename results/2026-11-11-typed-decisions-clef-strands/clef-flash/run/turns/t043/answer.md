The extension scored 76.20% on the reserved practice set, almost identical to the simpler repair’s 76.13%. I’ll compare their item-level effects and record this result before checking whether the added complexity helps on hidden validation.


```python
# Hypothesis: the gated extension's tuning gain may not generalize beyond the simpler two-neuron repair.
import hashlib, inspect, collections
def read_jsonl(path):
    return [json.loads(x) for x in open(path) if x.strip()]
r5 = read_jsonl(WORK/"gen_5_edited.jsonl")
r10 = read_jsonl(WORK/"gen_10_edited.jsonl")
print("Record keys:", list(r10[0]))
print("First score fields:", {k:v for k,v in r10[0].items() if k not in ("inputs","outputs","model_calls","calls") and len(str(v))<1500})
def rec_id(r):
    return r.get("id", r.get("example_id",r.get("problem_id")))
r5by = {rec_id(r):r for r in r5}
d = np.array([r["score"]-r5by[rec_id(r)]["score"] for r in r10])
print("Extension minus best_1:", dict(gain=d.mean(), se=d.std(ddof=1)/np.sqrt(len(d)), improved=int((d>1e-8).sum()), worsened=int((d<-1e-8).sum())))
print("Live arrays/functions:", [(k, tuple(v.shape) if hasattr(v,"shape") else "function") for k,v in list(globals().items()) if not k.startswith("_") and (isinstance(v,torch.Tensor) or inspect.isfunction(v))][-100:])
with open(WORK/"notes.md","a") as f:
    f.write(f"\n## Pass 10 — gated extension test\nOriginal 0.705333; extension 0.762000, gain +0.056667, paired SE 0.009925, n=300. Against best_1: gain {d.mean():+.6f}, paired SE {d.std(ddof=1)/np.sqrt(len(d)):.6f}; {int((d>1e-8).sum())} improved / {int((d<-1e-8).sum())} worsened. No evidence of incremental gain despite tune gain. Will check hidden validation before deciding; best_1 remains fallback.\n")
    f.write("\n## Planned pass 11 — targeted extension validation\nExact edits unchanged from pass 10: best_1 plus 22 gated/offset-compensated scorer neurons, tensors in targeted_candidate.pt. DECIDE both on all 300 hidden validation items; aggregate only. No further fitting.\n")
```
