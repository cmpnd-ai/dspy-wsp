The edited readout restores previously suppressed options: all three impact levels appear, billing is selected again, and sensor decisions no longer default to false. I’ll test a direct closed-form readout that targets correctness without first fitting away the old logits; this may reduce cancellation and improve stability.

```python
# Hypothesis: fitting centered correctness directly avoids residual old-logit leakage and yields a more stable readout.
rhs_direct=evecs.T@(X.T@yc[mask].astype(np.float64))
direct={}
for reg in [.001,.01,.1,1.,10.]:
    direct[reg]=(evecs@(rhs_direct/(evals+reg))).astype(np.float32)
for reg,w in direct.items():
    sf,qf=measure(A@w,rows,mask,y)
    st,qt=measure(At@w,rt,Mt,yt)
    diff=st-bt
    print(reg,"norm",round(np.linalg.norm(w),3),"fit",round(sf.mean(),4),"test",round(st.mean(),4),"gain",round(diff.mean(),4),"se",round(diff.std(ddof=1)/np.sqrt(420),4),qt)
```
