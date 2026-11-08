I’ll combine the two batches, reconstruct each record’s branch contribution, and solve sparse gain corrections under several declared regularization levels. I’ll use fit data only to choose a candidate, then test it on the untouched practice holdout.

```python
# Hypothesis: a larger causal dataset yields a more reliable sparse correction than the initial 200-case solve.
def readbatch(k,k1,k2):
    dr=work/f"fwd_{k}"
    mm=json.load(open(dr/"meta.json")); rr=json.load(open(dr/"requests.json"))
    zz=np.load(dr/"head_inputs.npy")[:,0,:].astype(float)/egnp
    z1=np.load(work/f"fwd_{k1}/head_inputs.npy")[:,0,:].astype(float)/egnp
    z2=np.load(work/f"fwd_{k2}/head_inputs.npy")[:,0,:].astype(float)/egnp
    a=s2*z1;b=-s1*z2;y=(s2-s1)*zz
    ab=(a*b).sum(1);a2=(a*a).sum(1);b2=(b*b).sum(1)
    ay=(a*y).sum(1);by=(b*y).sum(1);det=a2*b2-ab*ab
    c=(ay*b2-by*ab)/det;d=(by*a2-ay*ab)/det
    branch=((c[:,None]*z1-zz)*s1+(d[:,None]*z2-zz)*s2)/2
    ee=np.load(dr/"option_rows.npy").astype(float);ll=np.load(dr/"option_logits.npy").astype(float)
    mk=np.load(dr/"option_mask.npy");gg=np.zeros_like(ll)
    exs=[state_ex[statekey(r["state"]["inputs"]["state"])] for r in rr]
    for i,row in enumerate(mm["rows"]):
        g=exs[row["request"]]["labels"]["gold"][row["question"]]["probabilities"]
        for j,o in enumerate(row["options"]):gg[i,j]=g[o]
    return zz,branch,ee,ll,mk,gg,mm,exs
batch1=readbatch(2,3,4);batch2=readbatch(8,9,10)
Z0,B,E,L,mask,gold=[np.concatenate([batch1[j],batch2[j]],axis=0) for j in range(6)]
target=gold.argmax(1)
baseL=np.nan_to_num(L);EE=np.nan_to_num(E)
J=EE*egnp*B[:,None,:]-baseL[:,:,None]*(Z0*B)[:,None,:]/2048
J-= (J*mask[:,:,None]).sum(1,keepdims=True)/mask.sum(1)[:,None,None]
desired=temp*np.log(np.maximum(gold,.015))
desired-=(desired*mask).sum(1,keepdims=True)/mask.sum(1)[:,None]
basecenter=baseL-(baseL*mask).sum(1,keepdims=True)/mask.sum(1)[:,None]
X=J[mask];Y=(desired-basecenter)[mask]
sens=np.sqrt((X**2).mean(0));dims=np.argsort(sens)[-384:]
Xs=X[:,dims];gram=Xs.T@Xs;rhs=Xs.T@Y
sol2={}
print("base",np.mean(np.nanargmax(L,1)==target))
for ridge in [.1,1.,10.]:
    vv=np.zeros(2048);vv[dims]=np.linalg.solve(gram+ridge*np.eye(384),rhs);sol2[ridge]=vv
    for strength in [.5,1.,2.,3.]:
        print(ridge,strength,round(np.mean(np.nanargmax(logits_gain(vv*strength),1)==target),4))
v_initial=v.copy()
v2=sol2[1.]
np.save(work/"gain_direction_400.npy",v2)
save_edit(gn,g0*(1+torch.tensor(v2,dtype=torch.float32)))
note("Pass11: candidate B, ridge1 closed-form solve on 400 fit cases, 384 causally sensitive dimensions, strength1; DECIDE both all448 fit. No holdout used.")
```
