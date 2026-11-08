The audit read is complete. I’ll align each edited request with its cached original request, then compare the option margins for the controlled cue changes.
```python
# Hypothesis: exact request matching will show selective margin changes for the three repaired cues.
meta23=json.loads((work/"fwd_23"/"meta.json").read_text())
arr23={p.stem:np.load(p,mmap_mode="r") for p in (work/"fwd_23").glob("*.npy")}
print("AUDIT FILES",[p.name for p in (work/"fwd_23").iterdir()])
print("TAG VARIABLES",[(k,len(v)) for k,v in list(globals().items())
                        if isinstance(v,list) and "tag" in k.lower()])
print("TAG FILES",[p.name for p in work.glob("*tag*")])
print("ROW EXAMPLES",[meta23["rows"][i] for i in [0,320,640]])
print("REQUEST FILES",[(p.name,p.stat().st_size) for p in (work/"fwd_6").iterdir() if p.suffix==".json"])
for name in ["synthetic_tags","agent_tags","security_cf_tags"]:
    obj23=globals().get(name)
    if obj23 is not None:
        print(name, str(obj23[:8])[:1400])
```
