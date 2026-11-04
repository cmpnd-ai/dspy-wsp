```python
# Read the practice examples and print a small random sample to identify valid IDs and select diagnostic problems.
import json, random
p="/lambda/nfs/dspy-wsp-is/runs/hour/work/practice/trainset.jsonl"
rows=[json.loads(x) for x in open(p)]
random.seed(42)
sample=random.sample(rows, 8)
[(r["id"], r["inputs"].get("question","")[:180], r["labels"].get("answer","")) for r in sample]
```