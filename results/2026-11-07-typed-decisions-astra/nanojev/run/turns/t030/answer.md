The option-count adjustment does not improve development results. I’ll instead use the previously selected mixed soft/top-label contrast, refit it once on the 600 non-test records, and freeze it before testing on the reserved 300.
```python
# Hypothesis: a larger closed-form evidence estimate reduces variance while preserving the demonstrated scalar-readout mechanism.
# Hyperparameters fixed from fit450/dev150: mix3, ridge .003, strength .75; scale ridge with sample count.
D600=(Xc*RW[:,None,None])[mask]
Tmix=target_specs["mix3"]
y600=((Tmix-L0c)*RW[:,None])[mask]
D600t=torch.from_numpy(D600); y600t=torch.from_numpy(y600)
gram600=D600t.T@D600t; rhs600=D600t.T@y600t
lam600=.003*(600/450)
d_mix600=torch.linalg.solve(gram600+lam600*torch.eye(1024,dtype=torch.float64),rhs600).numpy()
torch.save(torch.tensor(d_mix600,dtype=torch.float32),candidate_dir/"mix600_direction.pt")
wmix600=set_scalar_direction(d_mix600,.75)
torch.save(wmix600,candidate_dir/"mix600_075_scalar.pt")
print("600-record correction norm",np.linalg.norm(d_mix600),"strength .75","fit-all score",
      metric_logits(head_logits(w0+.75*d_mix600),np.ones(len(rows),bool)))
note("Forward12 plan: paired DECIDE reserved test300, test_decide_request.json. Candidate mix600_075: closed-form centered target3*(.5gold+.5top-label), fit all600 fit+dev only; ridge=.004 (same per-item regularization as .003 on450); strength=.75 selected on earlier dev before refit. Exact edit scalar.weight=w_original+.75*d_mix600, saved candidate_weights/mix600_075_scalar.pt; all other tensors original. Best validated soft-full safely backed up best_1. No independent test labels used in this new solve or hyperparameter choice.")
```
