The failed block-23 edit is removed. A more localized descriptive signal appeared in block 3: MLP channel 382 was substantially more active near “Wait” than elsewhere in both transcripts. That does not prove it causes the loop, but it gives a better-defined candidate for a causal probe. I’ll zero only that channel’s corresponding block-3 down-projection column and compare against the original on the same four examples. The intervention will be kept only if the paired behavior improves. 

```python
# Causal probe a repeat-context candidate: zero only block-3 MLP down-projection column 382 and record this test.
import torch
from pathlib import Path
name="model.language_model.layers.3.mlp.down_proj.weight"
w=load_original(name).clone()
w[:,382]=0
save_edit(name,w)
base=Path("/lambda/nfs/dspy-wsp-is/runs/hour/work")
(base/"notes.md").write_text(
"Original diagnostic: 8/8 sampled items cut at 2048; repeated self-questioning and no boxed answer. Pass 3/4 block-23 channel 9,46 half/zero probes had paired gain 0 and all remained cut; that edit was removed.\n"
"Teacher-forced two transcripts: MLP activity near 'Wait' broad, but block 3 channel 382 had the largest consistently enhanced per-channel mean abs values (0.627 and 0.482 difference near vs far), motivating a single-channel causal probe. Correlational only.\n"
"Current edit: zero block-3 mlp.down_proj column 382 only. Pass 6 planned paired against original on train_5238/train_912/train_204/train_6074, max_tokens=2048.\n")
print("edit",name,"column 382 zeroed; relative norm change",float(torch.norm(w-load_original(name))/torch.norm(load_original(name))))
```