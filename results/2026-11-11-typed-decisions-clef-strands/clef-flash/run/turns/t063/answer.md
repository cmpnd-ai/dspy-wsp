I’ll verify the existing targeted-readout formulas before rebuilding them. The extension must retain the same three fields, gate threshold, regularization, and strength; only the amount of practice data should change.


```python
# Hypothesis: a like-for-like refit can isolate the value of extra examples without introducing a new tuned architecture.
qk=target_qkeys[0]
qd=qdata[qk]
print("Field ridge scaling:",qd["scale"],"trace/n",float(qd["xc"].square().sum()/qd["xc"].shape[0]),"trace/d",float(qd["xc"].square().sum()/qd["xc"].shape[1]))
print("Original target differences:", [(key,float(value[repair_mask].min()),float(value[repair_mask].max())) for key,value in target_sets.items()])
print("Original field input equality:",float((field_features-repair_features[:,0,:1024]).abs().max()))
print("Target soft relation to loggold:",float((target_sets["soft"]-center_options(torch.log(repair_gold.clamp_min(.015)),repair_mask))[repair_mask].abs().max()))
print("Class samples:",option_classes[:5],"selected",selected_classes,"expert gain",pair_gain,"full linear gain",full_gain)
print("Gate ridge scale:",gunit,float(GG.trace()/GG.shape[0]),"lambda",gate_lambda)
print("Reference gate accuracy:",float((gscores.argmax(-1)[repair_mask]==class_ids[repair_mask]).float().mean()))
print("Gate class coefficients reconstruction error:",float((gate_features@gcoef+gbias-gscores).abs().max()))
for c in selected_classes[:2]:
    rv,bv=expert_rows[c]
    print("Expert",option_classes[c],"row blocks",[float(rv[j:j+1024].norm()) for j in range(0,4096,1024)],"b",float(bv))
print("Candidate bias offsets:",[(int(i),float(repair_b2["tensors"][1][i])) for i in repair_b2["neurons"][2:6]])
```
