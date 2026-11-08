The readout exposes only the final normalized state—not the last MLP’s activations—so I won’t assume those activations are available. Instead, I can isolate the last residual write causally: perturb half of its post-FFN norm gains, use untouched coordinates as normalization anchors, then repeat for the other half. First I’ll collect the remaining unedited fitting and development states.
```python
# Hypothesis: selective post-FFN gain probes can identify the last residual contribution exactly from normalized head states.
PREFIX="model.language_model."
W2_NAME=PREFIX+"layers.23.feed_forward.w2.weight"
POST_NAME=PREFIX+"layers.23.post_ffn_norm.weight"
FINAL_NAME=PREFIX+"embedding_norm.weight"
gpost=load_original(POST_NAME).clone()
gn=load_original(FINAL_NAME).clone()
w2orig=load_original(W2_NAME).clone()
for name,g in [("post_ffn_norm",gpost),("embedding_norm",gn)]:
    a=g.numpy()
    print(name,"min/max",float(a.min()),float(a.max()),"abs quantiles",np.quantile(np.abs(a),[0,.1,.5,.9,1]))
print("w2 frobenius norm",float(w2orig.norm()),"row norm quantiles",np.quantile(w2orig.norm(dim=1).numpy(),[0,.1,.5,.9,1]))
unique_rows={}
for i,row in enumerate(meta2["rows"]):
    for j,op in enumerate(row["options"]):
        code=(row["type"],j)
        unique_rows.setdefault(code,collections.Counter())[e2[i,j].tobytes()]+=1
print("Distinct best forms by type/code:",[(k,len(v), sorted(v.values(),reverse=True)[:5]) for k,v in unique_rows.items()])
note("Head conventions: head_inputs is final normed answer-position state; hidden_mean is state-token average, NOT MLP activations. Plan: isolate last residual write via two complementary coordinate gain interventions at layers.23.post_ffn_norm. Untouched dimensions determine final-RMS scale ratio; perturbed dimensions then reveal their normalized last-write contribution. This avoids any backbone simulation.\n\nPlanned pass 3: READ original fit_b (180 balanced records, 900 questions), read_fit_b.json, layers 23/24. Current edit: none.")
```
