Removing narrative variation made the crown-jewel direction much more selective on discovery B. I’ll test these refined directions, including a low-impact readout for low-criticality, unprivileged assets. The strengths are set to offset the observed severity-margin deficits, not to sharpen all distributions.
I’ll install the refined security directions at a moderate dose and test them on discovery B. The payment and violation readouts stay unchanged.
```python
# Hypothesis: narrative-protected security readouts repair impact cues more selectively, including the neglected low-impact option.
security_low_pair=tuple(unused_neurons[6:8])
print("Low-impact pair:",security_low_pair,"original output weights",orig_sc3[0,list(security_low_pair)].tolist())
sec_specific_directions=[d_crown_specific,d_prod_specific,d_high_specific,d_low_specific]
sec_specific_pairs=security_neuron_pairs+[security_low_pair]
sec_specific_gains_moderate=[3.0,3.0,2.0,6.0]
sec_specific_cues_moderate=list(zip(sec_specific_pairs,sec_specific_directions,sec_specific_gains_moderate))
for label,direction in zip(["crown","production","highasset","lowimpact"],sec_specific_directions):
    np.save(WORK/f"security_{label}_specific.npy",direction)
install_linear_cues(base_cues+sec_specific_cues_moderate)
note("\n## Pass 17 planned: narrative-protected security cues, moderate dose\nExact edit: base payment4+violation2 plus security crown3,production3,highasset2,lowimpact6 on pairs "+str(sec_specific_pairs)+". Cue directions from same matched discovery-A counterfactuals, remove 128 unrelated-question PCs plus 48 within-condition narrative PCs and fixed nuisance means. Low-impact direction is low unprivileged minus medium unprivileged, contrast low-vs-medium, protected against privileged/high/crown means. "+json.dumps(edit_manifest())+
     "\nRead discovery B 200 cases/1000 questions via payment_read_b.json. Compare actual model output; no network simulation. Expect lower contamination of criticality readout by alert wording.")
print(edit_manifest())
```
