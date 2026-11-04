```python
# Compare completion tails across examples to check whether the capped generations share a self-checking loop.
import json
f="/lambda/nfs/dspy-wsp-is/runs/hour/work/gen_1.jsonl"
for line in open(f):
    r=json.loads(line)
    c=r["calls"][0]["completion"]
    print("\n",r["id"],"chars",len(c),"tail:",repr(c[-500:]))
```