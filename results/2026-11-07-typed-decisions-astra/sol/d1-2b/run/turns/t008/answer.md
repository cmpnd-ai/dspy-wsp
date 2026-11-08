The tool exposes final normalized answer states, but not the activations entering `w2`. That rules out an exact `w2` fit from these outputs. I’ll instead inspect the last post-FFN norm and test a sparse, dimension-specific gain correction. Its effect will need forward-pass confirmation because the residual path makes a head-only prediction approximate.

```python
# Hypothesis: anisotropic gains in the last post-FFN norm distort specific option contrasts.
prefix="model.language_model."
gn=prefix+"layers.23.post_ffn_norm.weight"
g0=load_original(gn)
eg=load_original(prefix+"embedding_norm.weight")
print("post FFN gain quantiles",np.quantile(g0.numpy(),[0,.01,.1,.5,.9,.99,1]))
print("embedding gain quantiles",np.quantile(eg.numpy(),[0,.01,.1,.5,.9,.99,1]))
print("norm tensors last",[(n,tuple(SHAPES[n])) for n in NAMES if 'layers.23' in n])
H=np.load(fdir/"head_inputs.npy")[:,0,:].astype(np.float64)
E=np.load(fdir/"option_rows.npy").astype(np.float64)
L=np.load(fdir/"option_logits.npy").astype(np.float64)
P=np.load(fdir/"probabilities.npy")
print("H rms",np.quantile(np.sqrt((H**2).mean(1)),[0,.5,1]))
print("state norm",np.quantile(np.sqrt((np.load(fdir/"hidden_mean.npy")**2).mean(2)),[0,.5,1]))
```
