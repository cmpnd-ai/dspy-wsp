I’ll now run the edited model on exactly the same probes. Comparing raw option margins will show whether the correction changes sensitivity to tool errors and policy violations.

```python
# Hypothesis: the edit's response to controlled state changes will confirm a cue-sensitive mechanism.
note("## Pass 8 complete / planned pass 9\nOriginal read completed for all 144 causal probe requests. Exact edit for pass 9 remains best_1: scorer neurons 791/1015, hard-margin ridge 1e-4, half strength, no other changes. READ edited on identical requests from causal_probes_edited.json; no labels or score used.")
print("Current edits:",edited())
print("Probe families:",dict(collections.Counter(m["family"] for m in probe_meta)))
```

I’ll read the edited model on the same 144 controlled requests, with the unchanged validation-confirmed weights.


```forward
{"file": "/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/typed-decisions/clef-flash-astra/work/causal_probes_edited.json"}
```
