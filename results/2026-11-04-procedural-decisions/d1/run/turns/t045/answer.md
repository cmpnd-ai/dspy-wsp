C is restored. I’ll write the next exact request to the notes and run the remaining practice partition. Those cases were used to fit C, so this measures consistency, not generalization.

```python
# Hypothesis: full C's gains are not confined to the partition used for earlier causal tests.
request={"type":"decide","model":"both","problems":fitids}
(work/"paired_fit_C.json").write_text(json.dumps(request))
with (work/"notes.md").open("a") as f:
    f.write("\nRestored full C, head.scorer.3.weight only. Next pass 17: paired 420 fitids (C-fit included), consistency diagnostic only; no selection on validation.\n")
print("C relative norm",np.linalg.norm(np.asarray(wC)-np.asarray(w0))/np.linalg.norm(np.asarray(w0)))
```
