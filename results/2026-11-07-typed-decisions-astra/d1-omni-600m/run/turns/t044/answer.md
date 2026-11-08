The previous inspection was too verbose. I’ll narrow it to the repair’s parameter layout and a compact accuracy table, keeping the reserved test examples untouched.
I’ll narrow the inspection to the repair’s parameter layout and a compact accuracy table. The reserved test examples will remain untouched.
```python
# Hypothesis: residual performance is concentrated in a few question types rather than a general calibration error.
print("D row:", D["rows"][0])
print("make_conditional dependencies:", make_conditional.__code__.co_names)
print("make_conditional constants:", [v for v in make_conditional.__code__.co_consts if isinstance(v, (float,int,str,tuple))])
print("make_conditional locals:", make_conditional.__code__.co_varnames)
print("Conditional solution structure:", type(conditional_solutions[qkeys[10]]))
csex = conditional_solutions[qkeys[10]]
print(str(csex)[:700])
print("Neurons: cumulative contrast RMS", [(n, float(contrib_rms[neuron_order[:n]].sum())) for n in [2,110,160,240]])
print("Question accuracy (%), original / shared / conditional:")
for qi,qk in enumerate(qkeys):
    s = D["dev"] & (qid==qi)
    goldidx = D["gold"][s].argmax(-1)
    accs = [float((ll[s].masked_fill(~D["mask"][s], -1e9).argmax(-1)==goldidx).float().mean()) for ll in [D["logits"],base_linear,cond_logits]]
    print("/".join(qk), [round(100*a,1) for a in accs])
print("Aggregate", evaluate_logits(cond_logits))
print("Scorer helpers", logits_scorer_params.__code__.co_names, scorer_features.__code__.co_names)
print("Conditional parameter shapes:", [tuple(v.shape) for v in cond_params])
```
