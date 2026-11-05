I’ll check what “head inputs” contains and verify how its rows map to questions before deriving an edit.

```python
# Hypothesis: the final scorer's contextual option features support a closed-form correction.
meta=json.load(open(work/"fwd_2/meta.json"))
print(str(meta)[:8500])
print("Scorer weights:", [(n,SHAPES[n]) for n in NAMES if "scorer" in n])
```
