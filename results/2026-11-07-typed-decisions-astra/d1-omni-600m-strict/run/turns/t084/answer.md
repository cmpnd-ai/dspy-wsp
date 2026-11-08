The repair gained **3.07 percentage points ± 0.75 paired SE** on confirmation cases. The gated security component alone added **1.02 ± 0.27 points** over the base repair. I’ll now submit these unchanged weights for hidden validation.
```python
# Hypothesis: the confirmation gain will carry over to hidden validation without any further tuning.
import hashlib
manifest21 = []
for name in sorted(edited()):
    t = load(name)
    o = load_original(name)
    manifest21.append(dict(name=name, changed=int((t != o).sum()),
                           sha256=hashlib.sha256(t.numpy().tobytes()).hexdigest()))
with open(work / "notes.md", "a") as f:
    f.write("\n## Pass 21 planned: hidden validation of gated security repair\n"
            "Exact edit: payment4+violation2 plus gated security crown3/production3/highasset2/lowimpact6; "
            "unchanged from passes18/20. "+json.dumps(manifest21)+"\n"
            "DECIDE both on all 300 hidden validation cases, aggregate results only. "
            "No tuning since independent confirmation.\n")
print(json.dumps(manifest21, indent=2))
```
