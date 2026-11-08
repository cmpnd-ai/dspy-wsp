The directions replicate on real discovery-B records: crown-jewel assets and production privileges have much larger projections on their respective directions. I’ll add low-strength readouts for these cues, preserving the already validated payment and violation repair.
```python
# Hypothesis: adding three protected security cue readouts will correct medium-severity defaults without broadly increasing severity.
for neuron,filename in [(594,"payment_direction.npy"),(518,"violation_direction_refined.npy")]:
    saved_d=np.load(WORK/filename)
    weight_dir=load("head.scorer.1.weight")[neuron].numpy()*np.asarray(ln_gain)
    print(neuron, "installed direction cosine",round(float(weight_dir@saved_d/(np.linalg.norm(weight_dir)*np.linalg.norm(saved_d))),6))
unused_neurons=[int(j) for j in np.argsort(np.abs(orig_sc3.numpy().ravel())) if int(j) not in set(cue_neurons+violation_neurons)]
security_neuron_pairs=[tuple(unused_neurons[i:i+2]) for i in range(0,6,2)]
print("Security neuron pairs:",security_neuron_pairs, "original output weights",orig_sc3[0,unused_neurons[:6]].tolist())
print("Installer loop structure:",[(x.opname,x.argval) for x in dis.get_instructions(install_linear_cues) if x.opname in ["UNPACK_SEQUENCE","FOR_ITER","STORE_FAST"]][:20])
np.save(WORK/"security_crown_direction.npy",d_crown)
np.save(WORK/"security_production_direction.npy",d_prod)
np.save(WORK/"security_highasset_direction.npy",d_high)
base_cues=[(*cue_neurons,np.load(WORK/"payment_direction.npy"),4.0),
           (*violation_neurons,np.load(WORK/"violation_direction_refined.npy"),2.0)]
sec_cues_low=[(*security_neuron_pairs[0],d_crown,2.0),
             (*security_neuron_pairs[1],d_prod,2.0),
             (*security_neuron_pairs[2],d_high,1.0)]
install_linear_cues(base_cues+sec_cues_low)
note("\n## Pass 15 planned: security impact readout, low dose\nExact edit: validated payment4+violation2 plus crown2,production2,highasset1 on scorer pairs "+str(security_neuron_pairs)+". Directions from matched discovery-A counterfactual contrasts, null-projected off 194-212 nuisance means/PCs; no fitting. Residual norms crown .03004, production .01264, highasset .01788. "+json.dumps(edit_manifest())+
     "\nRead same 200 discovery-B cases/1000 questions via payment_read_b.json. Predict selective security severity changes. Dose-response baseline is pass10 (base repair); confirmation records remain uninspected.")
print(edit_manifest())
```
This prepares the low-dose test; all model evaluation remains with the forward-pass tool.