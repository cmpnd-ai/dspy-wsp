Both probes are complete. I’ll now reconstruct the last branch contribution, check whether it reproduces the observed logits, and derive a regularized closed-form correction to selected gain dimensions using only the fit records.

```python
# Hypothesis: the measured last-branch contribution supports a sparse, option-specific gain repair.
egnp=eg.numpy().astype(np.float64)
Z0=H/egnp
Z1=np.load(work/"fwd_3/head_inputs.npy")[:,0,:].astype(np.float64)/egnp
Z2=np.load(work/"fwd_4/head_inputs.npy")[:,0,:].astype(np.float64)/egnp
s1=(d1/g0).numpy().astype(np.float64);s2=(d2/g0).numpy().astype(np.float64)
aa=s2*Z1;bb=-s1*Z2; yy=(s2-s1)*Z0
ab=(aa*bb).sum(1); a2=(aa*aa).sum(1);b2=(bb*bb).sum(1)
ay=(aa*yy).sum(1);by=(bb*yy).sum(1)
det=a2*b2-ab**2
c1=(ay*b2-by*ab)/det;c2=(by*a2-ay*ab)/det
B=((c1[:,None]*Z1-Z0)*s1+(c2[:,None]*Z2-Z0)*s2)/2
err=np.sqrt(((aa*c1[:,None]+bb*c2[:,None]-yy)**2).mean())
def logits_gain(v):
    z=Z0+B*v
    z=z/np.sqrt((z*z).mean(1,keepdims=True))
    return np.einsum("nd,nkd->nk",z*egnp,E)
mask=np.load(fdir/"option_mask.npy")
for s,k in [(s1,3),(s2,4)]:
    pred=logits_gain(s); obs=np.load(work/f"fwd_{k}/option_logits.npy")
    print("probe",k,"logit reconstruction rms",np.sqrt(np.mean((pred[mask]-obs[mask])**2)))
print("state reconstruction error",err,"scale ranges",np.quantile(c1,[0,.5,1]))
gold=np.zeros_like(L); target=[]
for i,row in enumerate(meta["rows"]):
    ex=readfit[row["request"]];g=ex["labels"]["gold"][row["question"]]["probabilities"]
    for j,o in enumerate(row["options"]):gold[i,j]=g[o]
    target.append(np.argmax(gold[i]))
target=np.array(target)
# Derivative includes the final RMS norm's radial projection.
EE=np.nan_to_num(E)
baseL=np.nan_to_num(L)
J=EE*egnp*B[:,None,:]-baseL[:,:,None]*(Z0*B)[:,None,:]/2048
J=J-(J*mask[:,:,None]).sum(1,keepdims=True)/mask.sum(1)[:,None,None]
temp=1.3374
desired=temp*np.log(np.maximum(gold,.015))
desired-= (desired*mask).sum(1,keepdims=True)/mask.sum(1)[:,None]
basecenter=baseL-(baseL*mask).sum(1,keepdims=True)/mask.sum(1)[:,None]
Y=(desired-basecenter)[mask]
X=J[mask]
# Select dimensions by causal logit sensitivity, not labels, then solve a ridge system once per declared penalty.
sens=np.sqrt((X*X).mean(0))
dims=np.argsort(sens)[-384:]
Xs=X[:,dims]
gram=Xs.T@Xs; rhs=Xs.T@Y
solutions={}
for ridge in [1.,10.,100.,1000.]:
    v=np.zeros(2048);v[dims]=np.linalg.solve(gram+ridge*np.eye(len(dims)),rhs)
    solutions[ridge]=v
    ll=logits_gain(v);acc=np.mean(np.nanargmax(ll,axis=1)==target)
    print("ridge",ridge,"fit acc",acc,"gain fractional norm",np.linalg.norm(v),"max",np.max(np.abs(v)))
drop_edit(gn)
note("Two causal probes reconstructed branch B; gain solutions use 384 largest causally sensitive dimensions, closed-form ridge vs gold log probabilities. Probes removed.")
```
