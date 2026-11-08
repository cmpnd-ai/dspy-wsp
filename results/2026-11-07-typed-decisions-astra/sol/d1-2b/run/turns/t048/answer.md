I’ll extend the paired readout audit to the next 150 holdout records, keeping the validated edit fixed.

```python
# Hypothesis: option-ranking changes persist across the second independent holdout slice.
with (work/"notes.md").open("a") as f:
    f.write("Pass21 READ original next150 pass17 holdout records; current edit unchanged locked C-strength2. Diagnostic only.\n")
```
