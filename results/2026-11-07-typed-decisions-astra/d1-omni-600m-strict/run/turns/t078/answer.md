The moderate gated edit is preferable: it has slightly better accuracy and substantially lower probability error than the strong dose. I’ll freeze that version and test it on the 450 confirmation cases; none of those records was used to build the security directions or gate.
```python
# Hypothesis: the moderate gated security repair generalizes better than the stronger dose, which overshoots low-impact decisions.
note("\nPass19 gated strong: discovery B .594 vs original .557, +.0370 SE .01071; increment vs base +.0060 SE .00446. Severity .50 versus moderate .52; severity probability squared error .19376 versus moderate .14215/base .16790. No unrelated ranking changes. Prefer moderate gated security (3,3,2,6) based on discovery only.")
for name in list(edited()):
    drop_edit(name)
for name in ["head.scorer.1.bias","head.scorer.1.weight","head.scorer.3.weight"]:
    save_edit(name,torch.load(WORK/"security_gated_moderate"/f"{name}.pt",map_location="cpu",weights_only=True))
(WORK/"confirm_security.json").write_text(json.dumps({"type":"decide","model":"both","problems":list(confirmation_ids)}))
note("\n## Pass 20 planned: independent practice confirmation of gated security extension\nExact edit restored security_gated_moderate: payment4+violation2 plus gated security crown3/production3/highasset2/lowimpact6. "+json.dumps(edit_manifest())+
     "\nDECIDE both on fixed 450 confirmation IDs, confirm_security.json. No case records or activations from confirmation used to derive directions, gate or doses; earlier base-repair aggregate was seen.")
print(edit_manifest())
```
The moderate gated checkpoint is now the candidate for confirmation.
