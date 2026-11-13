The reserved practice test confirms a 5.6-point gain, with paired SE 0.86 points: 91 records improved and 24 worsened. I’ll save this candidate and run the held-out validation set before treating it as a keeper.


```python
# Hypothesis: bypassing the scorer's compressed representation produces a generalizable gain rather than a fit-only effect.
import shutil, hashlib
def edit_fingerprint():
    ans={}
    for name in edited():
        t=load(name)
        ans[name]={"shape":list(t.shape),"sha256":hashlib.sha256(t.numpy().tobytes()).hexdigest()}
    return ans
candidate_dir = WORK/"candidate_linear_bypass"
if candidate_dir.exists():
    shutil.rmtree(candidate_dir)
shutil.copytree(WORK/"edited",candidate_dir)
print("Saved candidate:",candidate_dir)
print("Current edit fingerprints:",edit_fingerprint())
testorig=[json.loads(s) for s in (WORK/"gen_5_original.jsonl").read_text().splitlines()]
testedit=[json.loads(s) for s in (WORK/"gen_5_edited.jsonl").read_text().splitlines()]
bywf_test=collections.defaultdict(list)
for a,b in zip(testorig,testedit):
    assert a["id"]==b["id"]
    bywf_test[a["inputs"]["workflow"]].append((a["score"],b["score"]))
for wf,v in sorted(bywf_test.items()):
    v=np.array(v); d=v[:,1]-v[:,0]
    print(wf,"original/edited",v.mean(0).round(4),"gain",round(d.mean(),4),"SE",round(d.std(ddof=1)/np.sqrt(len(d)),4))
note("## Pass 5 result — reserved practice test\nn=300, original 0.705333, edited 0.761333; gain +0.056000, paired SE 0.008616; 91 improved, 24 worsened. Candidate backed up in candidate_linear_bypass/. Exact edits hash: "+json.dumps(edit_fingerprint())+".\n## Planned pass 6 — validation\nExact edit unchanged: two-neuron linear bypass (791,1015), strength 0.5, hard-margin ridge 1e-4 fit only on 500 practice records. DECIDE both on all 300 hidden validation items; aggregate only.")
```
