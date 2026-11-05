The edit improves every workflow, with the largest gain on sensor reports. I’ll inspect the saved features and cue-pair probes before testing how much of the gain depends on state evidence rather than option identity.
```python
# Hypothesis: subtracting the prior-only component should preserve gains if the repair reads state evidence.
print("allrows example:", str(allrows[0])[:3500])
print("probe pair example:",str(probe_pairs[0])[:1600])
print("probe requests example:",str(probe_requests[0])[:2000])
print("read dirs",[(x.name, sorted(y.name for y in x.iterdir())) for x in p.glob("fwd_*")][:3])
print("saved dirs",[x.name for x in p.iterdir() if x.is_dir()])
print("head weights norm", float(load_original("scalar.weight").norm()),"evidence diff",np.linalg.norm(delta_evidence-(delta900-delta_prior)))
print("test ids",len(test450),str(test450[:2]))
notes=p/"notes.md"
with notes.open("a") as f:
    f.write("\nPasses16-18 in-sample workflow gains: event .53333→.81778 (+.28444±.01945), multi-view .41333→.67111 (+.25778±.02119), sensors .49333→.92000 (+.42667±.03329). Current edit remains validated full900 ridge.1.\n")
```
