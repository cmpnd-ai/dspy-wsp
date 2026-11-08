I’ll assess the stronger attenuation, then compare the deadline direction’s projections across unrelated questions. That will show whether a scorer edit could strengthen deadline evidence without broadly favoring higher-risk answers.
The next analysis checks whether the stronger MLP intervention helped and whether the deadline direction is selective enough for a targeted scorer test.
```python
# Hypothesis: a state-contrast readout is more selective than attenuating broad MLP preference modes.
arr5,rowinfo5,meta5=load_read(5,read_ids_a)
read_summary(arr5,rowinfo5,arr2,rowinfo2)
deadline_summary(arr5,rowinfo5)
lc5=centered_options(np.asarray(arr5["option_logits"]),rowinfo5)
print("Logit corr/scale:",np.corrcoef(lc2,lc5)[0,1],float(lc2@lc5/(lc2@lc2)))
print("\nDeadline-direction effects across question families (geometric projections, not simulated scores)")
for key in sorted(groups):
    rr=[r for r in rowinfo3 if (r["wf"],r["q"])==key]
    spans=np.array([np.ptp(U3[r["i"],:len(r["opts"])]@d_deadline) for r in rr])
    gap=np.array([(U3[r["i"],r["gi"]]-U3[r["i"],r["pi"]])@d_deadline for r in rr if not r["ok"]])
    print(key,"span",round(float(spans.mean()),3),"error gap",round(float(gap.mean()),3))
clear_edits()
note("Pass 5 analyzed; head MLP attenuation experiment finished and restored to original. Moving to targeted scorer input-direction interventions. Deadline double-difference replicated on B: near +.0242 versus later -.0214 in centered-unit head features; corresponding original elevated-routine raw logit gaps remain -2.789/-2.947.")
```
