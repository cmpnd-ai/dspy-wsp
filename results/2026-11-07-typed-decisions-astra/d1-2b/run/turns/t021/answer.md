I’ll complete the odd-coordinate probe on the second fitting half. The intended repair will be restricted to the small subspace that distinguishes option tokens; all unrelated output directions can remain unchanged.
```python
# Hypothesis: the recovered final-write features can support a low-rank correction confined to option-token contrast directions.
probe8=read_arrays(8)
codd,scale8,err8=infer_subset(data["fit_a"],probe8,odd_idx)
cfull=np.zeros_like(data["fit_a"]["h"])
cfull[:,even_idx]=ceven
cfull[:,odd_idx]=codd
data["fit_a"]["c"]=cfull
data["fit_a"]["u"]=cfull/gpost.numpy()
t0=data["fit_a"]["h"]/gn.numpy()
print("Odd probe anchor error:",err8)
print("Final normalized residual RMS quantiles:",np.quantile(np.sqrt(np.mean(t0*t0,axis=1)),[0,.5,1]))
print("Last-write/final-residual norm ratio:",np.quantile(np.linalg.norm(cfull,axis=1)/np.linalg.norm(t0,axis=1),[0,.5,1]))
print("Pre-gain final-write RMS quantiles:",np.quantile(np.sqrt(np.mean(data["fit_a"]["u"]**2,axis=1)),[0,.5,1]))
note(f"Pass 8 odd anchor RMS error {err8:.3g}. Complementary probes recover final-write C=b/RMS(final residual), and U=C/post_ffn_gain. Any left-factor edit W2'=(I+B A)W2 changes the captured last-MLP write to U'=U+B A U, allowing head-only RMS/readout recomputation. B will be confined to option contrast directions, not arbitrary backbone directions.\n\nPlanned pass 9: READ edited fit_b (probe_odd_fit_b.json). Exact edit only post_ffn_norm.23 odd coordinates x1.5, even original.")
```
