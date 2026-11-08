The files provide one head-input vector per option, rather than explicit MLP activations. I’ll check their conventions and whether the option vectors share the same answer-position state; that will determine which closed-form intervention is actually supported by the observations.
```python
# Hypothesis: shared answer-position head inputs permit a norm-aware low-rank block-output intervention without simulating the backbone.
print(json.dumps({k:v for k,v in meta2.items() if k!="rows"},indent=2)[:8500])
req2 = json.loads((WORK/"fwd_2"/"requests.json").read_text())
print("Requests container:", type(req2), "length",len(req2))
print("First request:",json.dumps(req2[0] if isinstance(req2,list) else next(iter(req2.values())),indent=2)[:1800])
h2 = np.load(WORK/"fwd_2"/"head_inputs.npy")
e2 = np.load(WORK/"fwd_2"/"option_rows.npy")
mask2 = np.load(WORK/"fwd_2"/"option_mask.npy")
hm2 = np.load(WORK/"fwd_2"/"hidden_mean.npy")
print("Head-input same across options, first row:",[float(np.linalg.norm(h2[0,j]-h2[0,0])) for j in range(4)])
print("Norms head, option, hidden layers:",np.linalg.norm(h2[0,0]),np.linalg.norm(e2[0,0]),np.linalg.norm(hm2[0],axis=1))
print("Dot-product scores:",(h2[0,:4]*e2[0,:4]).sum(axis=1),np.load(WORK/"fwd_2"/"option_logits.npy")[0,:4])
print("Config:",[p.name for p in pathlib.Path("/Users/dbreunig/.cache/huggingface/hub/models--LiquidAI--d1-2B/snapshots/652f920e544ea37f81e4b0ce89b81d4ed0da90b1").glob("*config*")])
```
