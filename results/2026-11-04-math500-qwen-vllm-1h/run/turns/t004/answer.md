```python
# Inspect the practice set's IDs and sample questions to choose valid items for the diagnostic generation.
import json, random
p="/lambda/nfs/dspy-wsp-is/runs/hour/work/practice/trainset.jsonl"
rows=[json.loads(x) for x in open(p)]
random.seed(42)
sample=random.sample(rows, 8)
[(r["id"], r["inputs"].get("question","")[:180], r["labels"].get("answer","")) for r in sample]
```