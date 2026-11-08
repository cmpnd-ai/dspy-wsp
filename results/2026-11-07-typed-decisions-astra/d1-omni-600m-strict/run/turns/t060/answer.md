The installer expects each neuron pair as a nested tuple; the attempted call stopped before saving weights. I’ll correct that format and record the exact low-dose edit.
```python
# Hypothesis: low-dose, cue-specific security readouts will improve impact judgments while preserving the validated base repair.
base_cues=[(cue_neurons,np.load(WORK/"payment_direction.npy"),4.0),
           (violation_neurons,np.load(WORK/"violation_direction_refined.npy"),2.0)]
sec_cues_low=[(security_neuron_pairs[0],d_crown,2.0),
             (security_neuron_pairs[1],d_prod,2.0),
             (security_neuron_pairs[2],d_high,1.0)]
install_linear_cues(base_cues+sec_cues_low)
note("\n## Pass 15 planned: security impact readout, low dose\nExact edit: payment4+violation2 plus crown2,production2,highasset1 on scorer pairs "+str(security_neuron_pairs)+". Directions from matched discovery-A counterfactual contrasts, projected off 194-212 nuisance means/PCs. Residual norms crown .03004, production .01264, highasset .01788. "+json.dumps(edit_manifest())+
     "\nRead 200 discovery-B cases/1000 questions via payment_read_b.json. Predict selective security severity changes. Baseline for additional edit is pass10. No confirmation records used for directions.")
print(edit_manifest())
```
