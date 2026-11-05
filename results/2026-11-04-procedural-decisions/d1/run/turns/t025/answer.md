I’ll first score candidate B on the reserve features. Then I’ll derive a larger-sample closed-form readout from the 840 already explored cases, using the same regularization. The reserve cases will remain excluded from that derivation.

```python
# Hypothesis: more practice observations stabilize the same identified scorer direction without changing the repair mechanism.
rr,Ar,Lr,Mr,yr,reserve_order=readset(8,reserveids)
br,brq=measure(Lr,rr,Mr,yr)
sr,srq=measure(Ar@direct[.1],rr,Mr,yr)
print("B reserve",br.mean(),sr.mean(),"paired gain",np.mean(sr-br),"SE",np.std(sr-br,ddof=1)/np.sqrt(60),srq)
countt=Mt.sum(1)
Act=At-(At*Mt[:,:,None]).sum(1)[:,None,:]/countt[:,None,None]
yct=yt-yt.sum(1)[:,None]/countt[:,None]
Xall=np.concatenate([X,Act[Mt].astype(np.float64)])
yall=np.concatenate([yc[mask],yct[Mt]]).astype(np.float64)
# Fixed lambda=.1 from the earlier practice comparison; one direct linear solve.
wC=np.linalg.solve(Xall.T@Xall+.1*np.eye(1024),Xall.T@yall).astype(np.float32)
sc,scq=measure(Ar@wC,rr,Mr,yr)
print("C reserve",sc.mean(),"gain",np.mean(sc-br),"SE",np.std(sc-br,ddof=1)/np.sqrt(60),scq)
candidateC=torch.from_numpy(wC).reshape(1,-1)
save_edit("head.scorer.3.weight",candidateC)
torch.save(candidateC,work/"candidate_C.pt")
with open(work/"notes.md","a") as f:
    f.write(f"Candidate B reserve replay {sr.mean():.5f}, baseline {br.mean():.5f}. Candidate C same direct ridge .1 fitted pooled 840 explored cases, excludes 60 reserve; reserve replay {sc.mean():.5f}. Next paired actual 60 reserve with C, final scorer only.\n")
json.dump({"type":"decide","model":"both","problems":reserveids},open(work/"paired_reserve.json","w"))
```
