The extension improved hidden-validation accuracy to 77.07%, versus 70.20% originally—a gain of 6.87 ± 0.90 percentage points. I’ll preserve it as the best validated candidate, while noting that its advantage over the simpler repair is uncertain. Next I’ll inspect the remaining errors, especially the weak response to duplicate-invoice evidence.


```python
# Hypothesis: duplicate-invoice errors may reflect missing evidence upstream, rather than the final scoring bottleneck already repaired.
import shutil, collections
best2 = WORK/"best_2"
if best2.exists():
    shutil.rmtree(best2)
shutil.copytree(WORK/"edited",best2)
with open(WORK/"notes.md","a") as f:
    f.write("\n## Pass 11 — validation confirmed best_2\nOriginal 0.702000, extension 0.770667; gain +0.068667, paired SE 0.009000, n=300, 110 improved / 25 worsened. Backed up edited/ as best_2/. This has the best validation gain, but incremental gain over best_1 is uncertain (only +0.000667 ±0.005129 on reserved practice). Further work will not claim that incremental gain established.\n")
print("Read outputs:")
for p in sorted((WORK/"fwd_2").iterdir()):
    print(p.name, p.stat().st_size)
meta2 = json.load(open(WORK/"fwd_2"/"meta.json"))
print("Meta keys:", list(meta2))
for k,v in meta2.items():
    if k not in ["rows","questions","records"]:
        print(k, str(v)[:2500])
print("Variables:")
print([(k,type(v).__name__,tuple(v.shape) if hasattr(v,"shape") else (len(v) if isinstance(v,(dict,list,set)) else "")) for k,v in list(globals().items()) if not k.startswith("_")][:190])
print("Example:", str(r10[0]["calls"])[:3000])
```
