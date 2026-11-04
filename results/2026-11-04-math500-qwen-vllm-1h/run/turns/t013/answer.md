```python
# Inspect short excerpts from saved completions to identify the shared reason all diagnostic calls hit the token cap.
import json
f="/lambda/nfs/dspy-wsp-is/runs/hour/work/gen_1.jsonl"
for line in open(f):
    r=json.loads(line)
    print("ID",r.get("id"),"score",r.get("score"))
    print(json.dumps(r.get("outputs",r.get("calls",r)),ensure_ascii=False)[:900])
```