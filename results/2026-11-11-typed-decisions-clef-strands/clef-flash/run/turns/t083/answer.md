I’ll measure the original, shared-repair, and full-repair cue responses on the new seeds. This checks whether the earlier effects recur—especially tool errors shifting outcome toward partial success and constraint violations raising urgency—and keeps known limitations visible.


```python
# Hypothesis: the saved repair strengthens the same cue-sensitive margins on seeds excluded from its fit and tuning sets.
for _fname in ("read_probe_arrays", "simulate_probe_scoring", "probe_margin", "note", "edit_fingerprint"):
    try:
        print("\n" + _fname + ":\n" + inspect.getsource(globals()[_fname])[:4800])
    except (OSError, TypeError):
        print(_fname, "source unavailable")

_saved_best2 = torch.load(_handoff_work / "targeted_candidate.pt", weights_only=True)
_saved_best1 = torch.load(_handoff_work / "linear_bypass_candidate.pt", weights_only=True)
print("\nSaved best_2 tensor keys:", list(_saved_best2))
print("Exact saved/edit equality:", {n: bool(torch.equal(load(n), v.to(torch.bfloat16).float())) for n, v in _saved_best2.items()})
print("Probe families and levels:")
for _fam in sorted({m["family"] for m in held_probe_meta}):
    print(_fam, sorted({str(m["level"]) for m in held_probe_meta if m["family"] == _fam}))
print("Fingerprint:", edit_fingerprint())
```
