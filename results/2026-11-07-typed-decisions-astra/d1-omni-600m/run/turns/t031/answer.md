The question-specific results suggest another bottleneck: some state information may be shared across all option states and cancel in a shared readout. I’ll test option-conditional residual directions, which can use the same cue differently for “no discrepancy” and “material discrepancy.” This is still a direct, regularized solve on frozen activations.
```python
# Hypothesis: common state cues cancel in shared option contrasts; conditional readouts can recover them with option-specific signs.
conditional_solutions={}
conditional_logits={}
condrels=[0.01,0.03,0.1,0.3,1.0,3.0]
allcond={rel:torch.zeros_like(D["logits"]) for rel in condrels}
cond_stats=[]
for qno,qkey in enumerate(qkeys):
    fq=D["fit"]&(qid==qno)
    aq=qid==qno
    nopts=int(D["n"][aq][0])
    conditional_solutions[qkey]={}
    for opt in range(nopts):
        xf=D["x"][fq,opt].double()
        xmean=xf.mean(0); xc=xf-xmean
        yf=(target_mode-centered(base_linear))[fq,opt].double()
        ym=yf.mean(); yc=yf-ym
        K=xc@xc.T
        tr=float(K.trace()/len(xf))
        eo,Vo=torch.linalg.eigh(K)
        projected=Vo.T@yc
        sol={}
        for rel in condrels:
            lam=max(tr*rel,1e-8)
            dd=xc.T@(Vo@(projected/(eo+lam)))
            bias=ym-xmean@dd
            sol[rel]=(dd.float(),float(bias))
            allcond[rel][aq,opt]=D["x"][aq,opt]@dd.float()+float(bias)
        conditional_solutions[qkey][opt]=sol
for rel in condrels:
    ev=evaluate_logits(base_linear+allcond[rel])
    fit_ev=evaluate_logits(base_linear+allcond[rel],"fit")
    print("Option-conditional rel",rel,"fit acc",round(fit_ev["acc"],4),
          "dev",{k:round(v,5) for k,v in ev.items()})
print("Question effects for fixed rel .1:")
evc=evaluate_logits(base_linear+allcond[0.1],details=True)
for k,(orig,cur) in evc["by_question"].items():
    print(k[0][:8],k[1],"shared",round(qstats[k]["base"],3),"conditional",round(cur,3))
# Fit q/option gates, without answer labels.
qopt_keys=[(q,o) for q in range(len(qkeys)) for o in range(int(D["n"][qid==q][0]))]
qopt_index={k:j for j,k in enumerate(qopt_keys)}
ident=torch.zeros((NQ,5,len(qopt_keys)),dtype=torch.float32)-1
for j in range(NQ):
    for o in range(int(D["n"][j])):
        ident[j,o,qopt_index[int(qid[j]),o]]=1
idmean=(ident[D["fit"]]*D["mask"][D["fit"],:,None]/D["n"][D["fit"],None,None]).sum((0,1))/int(D["fit"].sum())
Yid=((ident-idmean)/D["n"].sqrt()[:,None,None])[fitmask].double()
idgateW=(Va@((Va.T@(A.T@Yid))/(ea[:,None]+float(ea.sum()/1024)*0.001))).float()
idgates=XA@idgateW+idmean
vld=D["dev"][:,None]&D["mask"]
gate_actual=torch.argmax(ident,dim=-1)
predgate=idgates.argmax(-1)
print("Q/option identity argmax accuracy on dev:",float((predgate[vld]==gate_actual[vld]).float().mean()))
print("Gate sign accuracy:",float((((idgates>0)==(ident>0))[vld]).float().mean()))
note("Exploratory mechanism: a shared linear residual cannot exploit state signal common to all option vectors; conditional q/option readouts allow different signs for that shared cue. Fit-only dual ridge solutions computed for fixed regularization grid; no edit installed, best_1 remains current. Gate fits use question/option identity only, not gold answers.")
```
The confirmed weights remain unchanged during this diagnostic.
