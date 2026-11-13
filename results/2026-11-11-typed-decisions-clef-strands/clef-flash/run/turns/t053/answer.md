Workflow-specific corrections did not beat the shared repair on tuning data, so I’m rejecting them. I’ll now separate the validated correction into its option, field–option product, and absolute-distance parts. This tests which interaction carries the useful evidence and whether one part is causing avoidable errors.


```python
# Hypothesis: the missing scoring direction is carried mainly by field-option interactions, not by an option-only preference shift.
repair_pair=list(repair_b1["pair"])
repair_block_base=[a.clone() for a in repair_current_tensors]
repair_block_stats=[]
repair_block_logs={}
repair_block_corrections=[]
rw,rb,ro=repair_base1_tensors
for start in (1024,2048,3072):
    correction=repair_gate*ro.flatten()[repair_pair[0]]*(repair_features[:,:,start:start+1024]@rw[repair_pair[0],start:start+1024])
    repair_block_corrections.append(center_options(correction,repair_mask))
print("Within-option correction RMS by block:",[float(x[repair_mask].square().mean().sqrt()) for x in repair_block_corrections])
repair_block_scales=[(0,1,1),(1,0,1),(1,1,0),(1,1,1),(.5,1,1),(1,.5,1),(1,1,.5),(1.5,1,1),(1,1.5,1),(1,1,1.5)]
for scales in repair_block_scales:
    tensors=[a.clone() for a in repair_current_tensors]
    for j,start in enumerate([1024,2048,3072]):
        tensors[0][repair_pair,start:start+1024]*=scales[j]
    tensors=tuple(a.bfloat16().float() for a in tensors)
    ll=repair_simulate(tensors)
    repair_block_logs[scales]=ll
    repair_block_stats.append((scales,repair_measure(ll,repair_fit)["acc"],repair_measure(ll,repair_tune),
                               repair_measure(ll,repair_tune,repair_current_logits)["gain"]))
print("Ablation/dose results (scales, fit acc, tune stats, gain over best2):")
for x in repair_block_stats: print(x)
with open(WORK/"notes.md","a") as f:
    f.write("\n## Workflow-specific corrections rejected\nFour separately fitted workflow corrections (one-shot centered ridge, 3 fixed regularizers, 3 strengths) reached at most 0.750 tune, versus 0.766 best_1 and 0.786 best_2. No weights saved. All local summaries now explicitly mask padded options; earlier unmasked per-field diagnostic table was invalid and is disregarded. Forward scores remain valid.\n")
    f.write("\n## Interaction-block causal ablation\nSeparating option-only, field*option, |field-option| components of the saved shared correction, retaining best_2 gates. Results: "+str(repair_block_stats)+"\n")
```
