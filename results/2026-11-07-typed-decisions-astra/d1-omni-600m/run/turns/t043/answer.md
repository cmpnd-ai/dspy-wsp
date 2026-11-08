I’ll check how much of each remaining error comes from the shared residual versus the question-specific channels. This distinguishes an under-read cue from interference introduced by the repair itself.
```python
# Hypothesis: the shared direction introduces question-specific interference that can be measured independently of the gated correction.
for name in ["evaluate_logits", "make_conditional", "logits_scorer_params", "scorer_features", "load_read_dataset"]:
    try:
        print("\n"+name+"\n"+inspect.getsource(globals()[name]))
    except OSError:
        import dis
        dis.dis(globals()[name])
print("D row", D["rows"][0])
print("Conditional solution example", {k: str(v)[:100] for k,v in conditional_solutions[qkeys[10]].items()})
print("Neurons: cumulative contrast RMS", [(n, float(contrib_rms[neuron_order[:n]].sum())) for n in [2,110,160,240]])
print("Question accuracies original/shared/conditional (dev):")
for qi,qk in enumerate(qkeys):
    s = D["dev"] & (qid==qi)
    goldidx = D["gold"][s].argmax(-1)
    accs = [float((ll[s].masked_fill(~D["mask"][s], -1e9).argmax(-1)==goldidx).float().mean()) for ll in [D["logits"],base_linear,cond_logits]]
    print(qk, [round(100*a,1) for a in accs])
```
