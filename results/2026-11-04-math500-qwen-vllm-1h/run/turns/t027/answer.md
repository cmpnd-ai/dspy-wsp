Halving channels 9 and 46 did not change the behavior on the four paired examples: both versions still hit the 512-token cap. That rules out this mild intervention as a useful repair. I’ll make a stronger causal probe by zeroing just those two block-23 MLP output directions and compare against the original on the same examples at the original 2,048-token cap. This tests whether their contribution matters at the full observed failure length; it is still a narrow, reversible edit. 

```python
# Strengthen the narrow causal probe by zeroing only block-23 MLP down-projection columns 9 and 46.
import torch
from pathlib import Path
name="model.language_model.layers.23.mlp.down_proj.weight"
w=load_original(name).clone()
w[:,[9,46]]=0
save_edit(name,w)
Path("/lambda/nfs/dspy-wsp-is/runs/hour/work/notes.md").write_text(
"Observed original failure: 8/8 cut at 2048 with repetitive self-checking; four additional examples all cut at 512.\n"
"Pass 3 paired test: half-strength block-23 down_proj columns 9,46 on four examples at 512; original and edited both scored 0/4 and all cut, paired gain 0, SE 0. Reject half-strength as ineffective.\n"
"Current probe: zero only columns 9,46 in model.language_model.layers.23.mlp.down_proj.weight (relative edit is narrow). Test versus original on the same four examples at 2048.\n")
print("zeroed candidate channels; changed edit relative to original:", float(torch.norm(w-load_original(name))/torch.norm(load_original(name))))
```