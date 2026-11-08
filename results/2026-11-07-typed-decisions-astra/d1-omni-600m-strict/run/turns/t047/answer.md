The valid head-input vectors are exactly unchanged by the scorer edit, so the observed ranking changes are localized to the four edited scorer neurons. I’ll restore the gain-2 checkpoint and record the exact confirmation test before running it.
```python
# Hypothesis: the four-neuron repair improves held-out practice decisions by reading existing state cues rather than changing encoder representations.
for name in list(edited()):
    drop_edit(name)
for name in ["head.scorer.1.bias","head.scorer.1.weight","head.scorer.3.weight"]:
    save_edit(name, torch.load(WORK/"payment4_violation2"/f"{name}.pt", map_location="cpu", weights_only=True))
confirmation_request = {"type":"decide","model":"both","problems":list(confirmation_ids)}
(WORK/"confirm_initial.json").write_text(json.dumps(confirmation_request))
note("\n## Pass 12 planned: independent practice confirmation\nExact edit: restored payment4_violation2 (payment +/-4 rows594/746, violation +/-2 rows518/33), all other tensors original. "+json.dumps(edit_manifest())+
     "\nDECIDE both on 450 predesignated confirmation IDs in confirm_initial.json; none used to derive/select directions. Read-pass causal features verified identical at valid option positions; previous False equality was NaN padding.")
print(edit_manifest())
print("Confirmation n:", len(confirmation_ids))
```
I’ll restore the gain-2 checkpoint and test it on the 450 predesignated confirmation cases. These cases were not used to derive the directions or choose their strengths.
```python
# Hypothesis: the four-neuron repair generalizes by reading existing state cues rather than changing encoder representations.
for name in list(edited()):
    drop_edit(name)
for name in ["head.scorer.1.bias","head.scorer.1.weight","head.scorer.3.weight"]:
    save_edit(name, torch.load(WORK/"payment4_violation2"/f"{name}.pt", map_location="cpu", weights_only=True))
confirmation_request = {"type":"decide","model":"both","problems":list(confirmation_ids)}
(WORK/"confirm_initial.json").write_text(json.dumps(confirmation_request))
note("\n## Pass 12 planned: independent practice confirmation\nExact edit: restored payment4_violation2 (payment +/-4 rows594/746, violation +/-2 rows518/33), all other tensors original. "+json.dumps(edit_manifest())+
     "\nDECIDE both on 450 predesignated confirmation IDs in confirm_initial.json; none used to derive/select directions. Read-pass causal features verified identical at valid option positions; previous False equality was NaN padding.")
print(edit_manifest())
print("Confirmation n:", len(confirmation_ids))
```
