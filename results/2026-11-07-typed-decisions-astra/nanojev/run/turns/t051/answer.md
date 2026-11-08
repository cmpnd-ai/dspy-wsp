Both components help, but the cross-fit results suggest that the state-varying component may benefit from more shrinkage. I’ll test two preselected candidates on validation: the best decomposed combination, and a modestly stronger version of the full direction. Neither changes another tensor.
```python
# Hypothesis: shrinking the component orthogonal to mean question/option features reduces variance without losing the useful readout repair.
def paired900_between(new_logits, old_logits):
    new_ok = (np.where(mask900,new_logits,-1e20).argmax(1)==G900.argmax(1)).astype(float)
    old_ok = (np.where(mask900,old_logits,-1e20).argmax(1)==G900.argmax(1)).astype(float)
    item_delta = np.array([(new_ok-old_ok)[ids900==i].mean() for i in unique900])
    return {"gain":float(item_delta.mean()),
            "se":float(item_delta.std(ddof=1)/np.sqrt(len(item_delta))),
            "improved":int((item_delta>0).sum()),"worsened":int((item_delta<0).sum())}
wdec900 = torch.from_numpy(w0+.875*dp900+.5*de900).reshape(1,-1)
w0875_900 = torch.from_numpy(w0+.875*d_mix900).reshape(1,-1)
torch.save(wdec900,work/"candidate_weights"/"mix900_decomposed_scalar.pt")
torch.save(w0875_900,work/"candidate_weights"/"mix900_0875_scalar.pt")
print("Cross-fit decomposed minus strength.75:",
      paired900_between(L900+.875*cv_prior_corr+.5*cv_evidence_corr,cross_logits))
print("Cross-fit strength.875 minus strength.75:",
      paired900_between(L900+.875*cross_correction,cross_logits))
print("Decomposed edit norm:",float(torch.linalg.vector_norm(wdec900-torch.from_numpy(w0).reshape(1,-1))))
note("\nTwo additional candidates frozen before validation: A=w0+.875*d_template900+.5*d_orthogonal900; B=w0+.875*d_mix900. Chosen from cached practice cross-fit only; small differences among them are not independent performance estimates. No further hyperparameters will be selected from validation.")
save_edit("scalar.weight",wdec900)
note("\nPlanned forward18: paired hidden VALIDATION300. Exact edit scalar.weight=candidate_weights/mix900_decomposed_scalar.pt =w0+.875*dp900+.5*de900; all other tensors original. Full900 practice fit; candidate A selected on cross-fit. best3 strength.75 preserved.")
```
