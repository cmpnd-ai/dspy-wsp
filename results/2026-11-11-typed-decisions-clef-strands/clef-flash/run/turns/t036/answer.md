I’ll log the exact unchanged edit before comparing it with the original model on the controlled probes. This comparison will test cue sensitivity, not accuracy on synthetic labels.

```python
# Hypothesis: the edit's response to controlled state changes will confirm a cue-sensitive mechanism.
note("## Pass 8 complete / planned pass 9\nOriginal read completed for all 144 causal probe requests. Exact edit for pass 9 remains best_1: scorer neurons 791/1015, hard-margin ridge 1e-4, half strength, no other changes. READ edited on identical requests from causal_probes_edited.json; no labels or score used.")
print("Current edits:",edited())
print("Probe families:",dict(collections.Counter(m["family"] for m in probe_meta)))
```
