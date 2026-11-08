The protected directions distinguish the intended cues while leaving unrelated question means largely unchanged. I’ll check that this separation also appears in discovery half B, then use nearly unused scorer neurons for a small causal intervention.
```python
# Hypothesis: protected security cue directions replicate on real discovery-B records and can be added with minimal loss of existing scorer capacity.
for label,direction,a,b in [("crown",d_crown,4,2),("prod",d_prod,3,2),("high",d_high,3,2)]:
    print("\nDirection",label)
    for crit in ["low","medium","high","crown_jewel"]:
        for priv in [False,True]:
            ix=[r["i"] for r in rowinfo3 if r["wf"]=="security_incidents" and r["q"]=="severity"
                and by_id[r["id"]]["inputs"]["state"]["context"]["asset_criticality"]==crit
                and ("deploy:production" in by_id[r["id"]]["inputs"]["state"]["principal"]["privileges"])==priv]
            if ix:
                vals=(U3[ix,a]-U3[ix,b])@direction
                print(crit,priv,"n",len(ix),"mean contrast projection",round(float(vals.mean()),4))
print("Already installed direction checks:")
for neuron,filename in [(594,"payment_direction.npy"),(518,"violation_direction_refined.npy")]:
    saved_d=np.load(WORK/filename)
    weight_dir=load("head.scorer.1.weight")[neuron].numpy()*ln_gain.numpy()
    print(neuron, "cos saved",float(weight_dir@saved_d/(np.linalg.norm(weight_dir)*np.linalg.norm(saved_d))))
unused_neurons=[int(j) for j in np.argsort(np.abs(orig_sc3.numpy().ravel())) if int(j) not in set(cue_neurons+violation_neurons)]
security_neuron_pairs=[tuple(unused_neurons[i:i+2]) for i in range(0,6,2)]
print("Security neuron pairs:",security_neuron_pairs, "original output weights",orig_sc3[0,unused_neurons[:6]].tolist())
print("Installer loop structure:",[(x.opname,x.argval) for x in dis.get_instructions(install_linear_cues) if x.opname in ["UNPACK_SEQUENCE","FOR_ITER","STORE_FAST"]][:25])
np.save(WORK/"security_crown_direction.npy",d_crown)
np.save(WORK/"security_production_direction.npy",d_prod)
np.save(WORK/"security_highasset_direction.npy",d_high)
```
