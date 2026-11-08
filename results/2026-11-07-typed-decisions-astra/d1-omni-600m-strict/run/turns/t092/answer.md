The gate separates agent records from other workflows in both discovery halves. I’ll apply it only to violation neurons 518 and 33, keeping the cue’s effective strength at 2. The prediction is specific: agent answers should stay unchanged, while unrelated answers lose the violation cue’s influence.
```python
# Hypothesis: gating the violation pair preserves its agent effects and suppresses off-workflow leakage.
agent_gate_scale22=32.0
agent_gate_threshold22=32.0*agent_gate_scale22*(
    agent_gate_ranges22[("A",False)][1]+agent_gate_ranges22[("A",True)][0])/2
w22=load("head.scorer.1.weight").clone()
b22=load("head.scorer.1.bias").clone()
out22=load("head.scorer.3.weight").clone()
for neuron,sign in [(518,1.0),(33,-1.0)]:
    direction22=agent_gate_scale22*np.asarray(d_agent_gate22)+sign*np.asarray(d_violation2)
    row22=direction22/ln_gain
    w22[neuron]=torch.as_tensor(row22,dtype=torch.float32)
    b22[neuron]=float(-np.dot(row22,ln_bias)-agent_gate_threshold22)
    out22[0,neuron]=sign
for name,t in [("head.scorer.1.weight",w22),("head.scorer.1.bias",b22),
               ("head.scorer.3.weight",out22)]:
    save_edit(name,t)
manifest22=edit_manifest()
with open(work/"notes.md","a") as f:
    f.write("\n## Pass 22 planned: causal workflow gate for violation readout\n"
            "Exact edits: best_2 unchanged except violation pair518/33 now "
            "(32*agent_gate +/- violation_direction)/LN_gain, bias cancels LN offset and subtracts "
            f"{agent_gate_threshold22:.9f}; output weights +/-1 preserve effective gain2 when active. "
            "Agent gate from discovery-A workflow means, nulling option/question means, other workflows, "
            "64 within-question PCs and violation direction. "+json.dumps(manifest22)+"\n"
            "READ edited on the same 200 discovery-B cases/1000 questions via payment_read_b.json. "
            "Prediction: agent logits nearly unchanged versus pass18, unrelated violation leakage removed. "
            "Gate scale/threshold chosen from discovery-A separation, no confirmation fitting.\n")
print("THRESHOLD",agent_gate_threshold22)
print("MANIFEST",manifest22)
```
