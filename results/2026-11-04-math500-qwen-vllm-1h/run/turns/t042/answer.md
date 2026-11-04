```python
# Hypothesis: inspect the paired completions to determine whether block-3 channel 382 reduced repetitive self-questioning or merely changed one answer by chance.
import json, os
base = "/lambda/nfs/dspy-wsp-is/runs/hour/work"
for fn in ("gen_6_original.jsonl", "gen_6_edited.jsonl"):
    print("\n", fn)
    with open(os.path.join(base, fn)) as f:
        for line in f:
            r = json.loads(line)
            print("id:", r.get("id"), "score:", r.get("score"))
            calls = r.get("calls", [])
            for c in calls:
                print("prompt:", str(c.get("prompt", ""))[-500:])
                print("completion:", str(c.get("completion", ""))[-900:])
```