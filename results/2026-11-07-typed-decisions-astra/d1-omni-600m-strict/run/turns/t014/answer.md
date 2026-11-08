I’ll check replication of the deadline direction and inspect the head’s projection weights for dominant directions. This will help distinguish a missing scorer readout from the head amplifying a broad option-preference signal at the expense of state evidence.
```python
# Hypothesis: head amplification of a few broad preference directions suppresses a reproducible, state-sensitive deadline signal.
def load_read(k, valid_ids):
    folder=WORK/f"fwd_{k}"
    meta=json.loads((folder/"meta.json").read_text())
    reqs=json.loads((folder/"requests.json").read_text())
    reqids=[request_to_id[canonical(req)] for req in reqs]
    assert set(reqids)==set(valid_ids)
    arrays={p.stem:np.load(p,mmap_mode="r") for p in folder.glob("*.npy")}
    info=[]
    for i,m in enumerate(meta["rows"]):
        eid=reqids[m["request"]]
        gg=by_id[eid]["labels"]["gold"][m["question"]]["probabilities"]
        g=np.array([gg[o] for o in m["options"]])
        pp=arrays["probabilities"][i,:len(g)]
        info.append(dict(i=i,id=eid,wf=by_id[eid]["inputs"]["workflow"],q=m["question"],typ=m["type"],
                         opts=m["options"],g=g,gi=int(g.argmax()),pi=int(pp.argmax()),ok=g.argmax()==pp.argmax()))
    return arrays,info,meta
arr3,rowinfo3,meta3=load_read(3,read_ids_b)
U3=unit_center_features(np.asarray(arr3["head_inputs"]))
TU3=unit_center_features(np.asarray(arr3["trunk_markers"]))
for name,aa,info,uf in [("A",arr2,rowinfo2,U2),("B",arr3,rowinfo3,U3)]:
    rs=[r for r in info if r["wf"]=="invoice_processing" and r["q"]=="urgency"]
    for flag in [False,True]:
        subset=[r for r in rs if (by_id[r["id"]]["inputs"]["state"]["payment"]["days_until_due"]<=3)==flag]
        print(name,"near",flag,"n",len(subset),"deadline projection gap",round(float(np.mean([(uf[r["i"],2]-uf[r["i"],1])@d_deadline for r in subset])),4),
              "logit elevated-routine",round(float(np.mean([aa["option_logits"][r["i"],2]-aa["option_logits"][r["i"],1] for r in subset])),3))
# Weight-only spectra; no network evaluation.
head_weight_cache={}
head_svd={}
for layer in [0,1]:
    for kind in ["linear2.weight","self_attn.out_proj.weight"]:
        n=f"head.head.layers.{layer}.{kind}"
        w=load_original(n)
        head_weight_cache[n]=w
        u,s,vh=torch.linalg.svd(w,full_matrices=False)
        head_svd[n]=(u[:,:12].clone(),s[:12].clone(),vh[:12].clone())
        print(n,"top singular values",np.round(s[:8].numpy(),2),"fraction top4",float(s[:4].square().sum()/s.square().sum()))
HH=np.asarray(arr2["head_inputs"])[valid2]
TT=np.asarray(arr2["trunk_markers"])[valid2]
delta_HT=HH-TT
_,sh,vh=np.linalg.svd(delta_HT-delta_HT.mean(0),full_matrices=False)
print("Head-minus-trunk variance top PCs",np.round(sh[:10]**2/np.sum(sh**2),3))
head_pc=vh[:10].copy()
for n,(u,s,vh_small) in head_svd.items():
    print(n,"top mode vs head-change PCs",np.round(np.max(np.abs(head_pc@u.numpy()),axis=0)[:6],3))
note("Pass 3 read complete. Examining replication of invoice deadline direction and weight-only singular modes. No edits yet; candidate mechanisms not accepted without causal interventions.")
```
