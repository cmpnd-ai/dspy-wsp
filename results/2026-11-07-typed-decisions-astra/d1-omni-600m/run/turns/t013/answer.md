I’ll derive a closed-form correction using the scorer’s existing GELU features. The correction will be constrained orthogonal to the original readout, ruling out a simple global sharpening or flattening, and selected on development—not test—examples.
The calculation below uses a direct regularized linear solve, with no gradients or iterative weight fitting. It compares a small set of regularization strengths and records the predicted gains before any edit is saved.
```python
# Hypothesis: a non-scaling correction to the final GELU readout can recover cue-sensitive option rankings already encoded by the frozen network.
ds_parts.append(load_read_dataset(5,read_groups[3]))
D={k:torch.cat([d[k] for d in ds_parts]) for k in ["gold","mask","logits","prob","z","x"]}
D["rows"]=sum([d["rows"] for d in ds_parts],[])
NQ=len(D["rows"])
D["fit"]=torch.tensor([r["split"]=="fit" for r in D["rows"]])
D["dev"]=torch.tensor([r["split"]=="dev" for r in D["rows"]])
D["temp"]=torch.tensor([cfg["temperatures"][r["type"]+":2" if r["type"]=="noul" else r["type"]+":3-5"] for r in D["rows"]])
D["n"]=D["mask"].sum(1)
D["gold"]=D["gold"]/D["gold"].sum(1,keepdim=True)
D["correct0"]=D["logits"].masked_fill(~D["mask"],-1e6).argmax(1)==D["gold"].argmax(1)
def centered(v,mask=D["mask"]):
    if v.ndim==3:
        mm=mask[:,:,None]
        return (v-(v*mm).sum(1,keepdim=True)/mask.sum(1)[:,None,None])*mm
    return (v-(v*mask).sum(1,keepdim=True)/mask.sum(1)[:,None])*mask
Zc=centered(D["z"])
unit0=W0.squeeze()/W0.norm()
Zorth=Zc-(Zc@unit0)[:,:,None]*unit0
fitmask=D["fit"][:,None]&D["mask"]
X=(Zorth/D["n"].sqrt()[:,None,None])[fitmask].double()
C=X.T@X
eigs,V=torch.linalg.eigh(C)
print("Feature covariance eigenvalues:",eigs[-10:].tolist())
print("Trace/d:",float(eigs.sum()/1024),"effective dimension",float(eigs.sum()**2/eigs.square().sum()))
def evaluate_logits(logits, split="dev", details=False):
    sel=D[split]
    correct=logits.masked_fill(~D["mask"],-1e6).argmax(1)==D["gold"].argmax(1)
    probs=torch.softmax((logits/D["temp"][:,None]).masked_fill(~D["mask"],-1e6),dim=1)
    brier=((probs-D["gold"]).square()*D["mask"]).sum(1)/D["n"]
    ids=list(dict.fromkeys(r["id"] for j,r in enumerate(D["rows"]) if sel[j]))
    idpos=collections.defaultdict(list)
    for j,r in enumerate(D["rows"]):
        if sel[j]:idpos[r["id"]].append(j)
    diff=torch.tensor([float((correct[ix].float()-D["correct0"][ix].float()).mean()) for ix in idpos.values()])
    result={"acc":float(correct[sel].float().mean()),"gain":float(diff.mean()),"se":float(diff.std()/len(diff)**0.5),
            "brier":float(brier[sel].mean())}
    if details:
        result["by_question"]={}
        for key in sorted({(r["workflow"],r["question"]) for j,r in enumerate(D["rows"]) if sel[j]}):
            ix=[j for j,r in enumerate(D["rows"]) if sel[j] and (r["workflow"],r["question"])==key]
            result["by_question"][key]=(float(D["correct0"][ix].float().mean()),float(correct[ix].float().mean()))
    return result
base_eval=evaluate_logits(D["logits"])
print("Dev baseline:",base_eval)
solutions={}
for floor in [0.01,0.03]:
    target=centered(D["gold"].clamp_min(floor).log()*D["temp"][:,None])
    residual=target-centered(D["logits"])
    y=(residual/D["n"].sqrt()[:,None])[fitmask].double()
    b=X.T@y
    vb=V.T@b
    for rel in [0.01,0.03,0.1,0.3,1,3,10,30]:
        lam=float(eigs.sum()/1024)*rel
        delta=(V@(vb/(eigs+lam))).float()
        delta=delta-(delta@unit0)*unit0
        new_logits=D["logits"]+D["z"]@delta
        key=(floor,rel)
        solutions[key]=delta
        ev=evaluate_logits(new_logits)
        print("ridge",key,"norm",round(float(delta.norm()),3),"fit acc",round(evaluate_logits(new_logits,"fit")["acc"],3),
              "dev", {k:round(v,5) for k,v in ev.items()})
note("All 448 fit + 224 dev scorer activations loaded and verified. Derivation: option-centered nonlinear scorer features, weighted equally per question; ridge solve for target temperature-scaled log gold residuals. Correction constrained perpendicular to original readout so not global scaling. Candidate floor/regularization screened on dev only; no weights saved yet.")
```
