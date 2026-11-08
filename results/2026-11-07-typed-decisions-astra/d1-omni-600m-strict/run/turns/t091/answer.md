The violation cue improved agent outcome, risk and urgency, but also changed unrelated invoice and security answers. I’ll derive an agent-workflow gate from discovery-A feature geometry and test whether it preserves the intended benefit while removing that leakage. The payment and security repairs will stay fixed.
```python
# Hypothesis: agent-workflow identity is geometrically separable, so the violation readout can be restricted without weakening its intended signal.
import torch
from collections import defaultdict
gate_groups22 = defaultdict(list)
for r in rowinfo2:
    gate_groups22[(r["wf"],r["q"])].append(r["i"])
agent_option_means22 = []
other_option_means22 = []
within_question22 = []
for (wf,q), idx in sorted(gate_groups22.items()):
    nopt = len(rowinfo2[idx[0]]["opts"])
    f = np.asarray(U2[idx,:nopt], dtype=np.float64)
    m = f.mean(axis=0)
    (agent_option_means22 if wf=="agent_trace_observability" else other_option_means22).extend(m)
    within_question22.append((f-m[None,:,:]).reshape(-1,1024))
agent_mean22 = np.mean(agent_option_means22,axis=0)
within_matrix22 = np.concatenate(within_question22)
_,_,within_vh22 = np.linalg.svd(within_matrix22,full_matrices=False)
agent_nuisance22 = np.concatenate([
    np.array(other_option_means22),
    np.array(agent_option_means22)-agent_mean22,
    within_vh22[:64],
    np.asarray(d_violation2).reshape(1,-1),
    np.ones((1,1024))/32,
],axis=0)
d_agent_gate22, agent_gate_resnorm22, agent_gate_rank22 = null_project_direction(agent_mean22,agent_nuisance22)
print("GATE RESIDUAL",agent_gate_resnorm22,"NUISANCE RANK",agent_gate_rank22)
agent_gate_ranges22={}
for tag,U,rows in [("A",U2,rowinfo2),("B",U3,rowinfo3)]:
    for target in [False,True]:
        vals22=np.concatenate([U[r["i"],:len(r["opts"])]@d_agent_gate22 for r in rows
                               if (r["wf"]=="agent_trace_observability")==target])
        agent_gate_ranges22[(tag,target)]=(float(vals22.min()),float(vals22.max()))
        print(tag, "agent" if target else "other", 
              np.round(np.quantile(vals22,[0,.01,.5,.99,1]),5))
print("CUE ROW CHECK",
      float(torch.linalg.vector_norm(load("head.scorer.1.weight")[518] -
            torch.tensor(np.asarray(d_violation2)/ln_gain,dtype=torch.float32))))
np.save(work/"agent_workflow_gate.npy",d_agent_gate22)
```
