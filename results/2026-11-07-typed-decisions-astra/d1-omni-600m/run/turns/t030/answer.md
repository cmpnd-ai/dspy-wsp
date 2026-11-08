Workflow-specific readouts offer only small gains, so I won’t add that complexity. I’ll test a narrower hypothesis: the scorer needs question-specific interpretations of the same state—for example, a constraint violation affects outcome, risk, and urgency differently. I’ll check whether those distinctions are recoverable from the frozen scorer inputs.
```python
# Hypothesis: question-specific residual directions resolve semantic interference that a single shared scorer direction cannot express.
qkeys=sorted({(r["workflow"],r["question"]) for r in D["rows"]})
qid=torch.tensor([qkeys.index((r["workflow"],r["question"])) for r in D["rows"]])
qrels=[0.01,0.03,0.1,0.3,1,3]
qsolutions={}
qstats={}
base_linear_ev=evaluate_logits(base_linear)
for qno,qkey in enumerate(qkeys):
    fm=fitmask&(qid==qno)[:,None]
    xq=(Xc/D["n"].sqrt()[:,None,None])[fm].double()
    yq=((target_mode-centered(base_linear))/D["n"].sqrt()[:,None])[fm].double()
    cq=xq.T@xq
    eq,vq=torch.linalg.eigh(cq)
    bq=vq.T@(xq.T@yq)
    ds={}
    sel=D["dev"]&(qid==qno)
    fitq=D["fit"]&(qid==qno)
    z0=base_linear[sel].masked_fill(~D["mask"][sel],-1e6)
    g=D["gold"][sel].argmax(1)
    bacc=float((z0.argmax(1)==g).float().mean())
    scores=[]
    for rel in qrels:
        dq=(vq@(bq/(eq+float(eq.sum()/1024)*rel))).float()
        ds[rel]=dq
        l=z0+D["x"][sel]@dq
        acc=float((l.argmax(1)==g).float().mean())
        scores.append(acc)
    qsolutions[qkey]=ds
    qstats[qkey]=dict(base=bacc,accs=scores)
    print(qkey[0][:8],qkey[1],"base",round(bacc,3),"q-specific",[round(a,3) for a in scores])
for rel in qrels:
    add=torch.stack([D["x"][j]@qsolutions[(r["workflow"],r["question"])][rel] for j,r in enumerate(D["rows"])])
    ev=evaluate_logits(base_linear+add)
    print("All question directions rel",rel,{k:round(v,5) for k,v in ev.items()})
# Gold-free identity gates ensure any conditional edit can be implemented using only existing hidden states.
qlab=2*F.one_hot(qid,len(qkeys)).float()-1
qmean=qlab[D["fit"]].mean(0)
Yqgate=((qlab-qmean)[:,None,:].expand(-1,5,-1)/D["n"].sqrt()[:,None,None])[fitmask].double()
qgateW=(Va@((Va.T@(A.T@Yqgate))/(ea[:,None]+float(ea.sum()/1024)*0.003))).float()
qgates=XA@qgateW+qmean
valid=D["dev"][:,None]&D["mask"]
wrong=0; minpos=10.; maxneg=-10.
for q in range(len(qkeys)):
    pos=valid&(qid==q)[:,None]; neg=valid&(qid!=q)[:,None]
    wrong+=int((qgates[:,:,q][pos]<0).sum())+int((qgates[:,:,q][neg]>0).sum())
    minpos=min(minpos,float(qgates[:,:,q][pos].min()))
    maxneg=max(maxneg,float(qgates[:,:,q][neg].max()))
print("Question identity gates dev sign errors",wrong,"min positive",minpos,"max negative",maxneg)
note("Workflow-level gated readouts had only small dev benefits and are not installed. Testing question-level interference via independent fit-only ridge residuals. Identity gates use workflow/question labels only (no gold answer labels); no current weight changes. Validated best_1 remains installed.")
```
The confirmed repair remains installed. This analysis will only propose a further edit if question-specific readouts show a substantial development gain and their gates can be recovered reliably from existing activations.