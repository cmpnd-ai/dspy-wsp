The valid head-input vectors are unchanged, localizing the effect to the four edited scorer neurons. I’ll restore violation gain 2 and prepare the independent confirmation test.
```python
# Hypothesis: payment gain4 plus violation gain2 generalizes better than the higher-dose violation edit.
for name in list(edited()):
    drop_edit(name)
for name in ["head.scorer.1.bias", "head.scorer.1.weight", "head.scorer.3.weight"]:
    save_edit(name, torch.load(WORK/"payment4_violation2"/f"{name}.pt", map_location="cpu", weights_only=True))
confirmation_request = {"type": "decide", "model": "both", "problems": list(confirmation_ids)}
(WORK/"confirm_initial.json").write_text(json.dumps(confirmation_request))
note("\n## Pass 12 planned: independent practice confirmation\nExact edit: restored payment4_violation2 (payment +/-4 rows594/746, violation +/-2 rows518/33), all other tensors original. "+json.dumps(edit_manifest())+
     "\nDECIDE both on 450 predesignated confirmation IDs in confirm_initial.json; none used to derive/select directions. Valid exported head features are bit-identical; previous equality failure was NaN padding.")
print(edit_manifest())
print("Confirmation n:", len(confirmation_ids))
```
