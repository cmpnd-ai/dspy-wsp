The refinement is validation-confirmed at 73.4%, a 15.2-point gain with paired SE 1.40 points. I’ll back it up and investigate one remaining limitation: a single shared readout direction may mix evidence that should be interpreted differently across question types.

```python
# Hypothesis: residual errors arise partly because a rank-one key repair shares one evidence direction across different question contexts.
lognote("Pass 14 refined rank1 hidden validation: .582000 -> .734000, gain .152000, "
        "paired SE .014019; 174 improved, 35 worsened. Best confirmed score improves over best_1.")
BEST2=WORK/"best_2"
BEST2.mkdir(exist_ok=True)
shutil.copytree(WORK/"edited",BEST2/"edited",dirs_exist_ok=True)
torch.save({"k.weight":load("k.weight"),"u":u,"v":v_ref},BEST2/"weights_and_direction.pt")
json.dump({"validation_gain":.152,"validation_se":.014018939251885253,
           "test_gain":.15666666666666668,"test_se":.01938053699340398,
           "formula":"K0 + outer(u,v_ref), gold floor .1, ridge .0003, strength .75, fit=540"},
          open(BEST2/"results.json","w"))
Qorth=fit["Q"]-(fit["Q"]@u)[:,None]*u
qoeig,qovec=torch.linalg.eigh(Qorth.T.double()@Qorth.double()/len(Qorth))
OP=center_options(fit["o"],fit["mask"])[fit["mask"]]
opeig,opvec=torch.linalg.eigh(OP.T.double()@OP.double()/len(OP))
total_qvar=((fit["Q"]-qmean)**2).sum().item()
for grouping,label in [([g.split("/")[0] for g in fit["group"]],"workflow"),
                       ([m["type"] for m in fit["meta"]],"question type"),
                       (fit["group"],"question schema")]:
    within=0.
    for g in sorted(set(grouping)):
        ix=torch.tensor([i for i,x in enumerate(grouping) if x==g])
        xx=fit["Q"][ix]
        within+=((xx-xx.mean(0))**2).sum().item()
    print(label,"explained query variance",round(1-within/total_qvar,4))
print("Residual-query variance captured:",
      {r:round((qoeig[-r:].sum()/qoeig.sum()).item(),4) for r in [2,4,8,16]})
print("Option-contrast variance captured:",
      {r:round((opeig[-r:].sum()/opeig.sum()).item(),4) for r in [32,64,128,256]})
print("Best2 development remaining accuracy by question:")
group_report(dev,dev["L"]+Cdev_ref)
lognote("Current best confirmed edit backed up at best_2/edited. Investigate rank>1 key repair only if fit/dev "
        "supports it: residual query modes may condition the option-evidence readout by question context.")
```
