The full repair helps all three workflows. In the example above, it corrects both a stale owner choice and a severity error. Next I’ll restore C—the best validation-confirmed edit—and measure it on the other 420 practice cases, keeping that result explicitly separate from held-out evidence.

```python
# Hypothesis: the confirmed readout repair should improve all workflows in the other fitting partition as well.
import inspect
print(inspect.getsource(measure))
print(inspect.getsource(readset))
print("list variables", [(k,len(v)) for k,v in list(globals().items()) if isinstance(v,list) and len(v)>50])
save_edit("head.scorer.3.weight", candidateC)
req=json.loads((work/"read_fit.json").read_text())
print("read fit request keys",req.keys())
print(str(req)[:400])
print("C edit relative norm",float(torch.linalg.vector_norm(wC-w0)/torch.linalg.vector_norm(w0)))
```
