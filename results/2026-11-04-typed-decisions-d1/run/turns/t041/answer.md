I’ll recover the cached experiment variables and inspect the recorded diagnostics before completing the remaining causal tests. The validated full-strength edit remains the fallback.

```python
# Hypothesis: the saved readout correction repairs contextual discrimination, with an intermediate optimal strength.
print("Relevant variables:", [k for k in globals() if not k.startswith("_") and any(s in k.lower() for s in ["delta", "test", "fit", "path", "work", "meta", "dose", "feature", "req"])])
print("Current edits:", edited())
from pathlib import Path
work = Path("/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/typed-decisions/d1/work")
print("Files:", sorted(p.name for p in work.iterdir()))
```
