The actual program matches the conditional scorer prediction: 70.09% development accuracy, +15.09 points over original with paired SE 1.71. I’ll record its incremental effect and prepare a hidden-validation check.
```python
# Hypothesis: the option-conditional extension improves generalization by reading common state cues rather than memorizing fit examples.
ed12=[json.loads(s) for s in (WORK/"gen_12_edited.jsonl").read_text().splitlines()]
ed12_by_id={str(r["id"]):r for r in ed12}
inc=np.array([ed12_by_id[i]["score"]-ed8_by_id[i]["score"] for i in splits["dev"]])
print("Conditional vs shared on dev: gain",inc.mean(),"SE",inc.std(ddof=1)/np.sqrt(len(inc)),
      "improved",sum(inc>0),"worsened",sum(inc<0))
note("Pass 12 actual conditional scorer: dev original55.000% ->70.089%, +15.089pp SE1.706pp, 135 improved/32 worsened. Exact local prediction confirmed. Increment over shared repair: "+str(inc.mean())+" SE "+str(inc.std(ddof=1)/np.sqrt(len(inc)))+
     ".\nNext pass: VALIDATION both on300 hidden examples. Exact edits: conditional_candidate.pt (110 scorer neurons modified; base pair +54 option-gated directions on14 baseline-poor questions). No new fitting or changes since pass12.")
```
The confirmed two-neuron repair remains backed up; this check will determine whether the larger conditional edit is worth keeping.
