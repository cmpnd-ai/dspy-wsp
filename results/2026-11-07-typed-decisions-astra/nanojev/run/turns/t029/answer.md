The cue tests show real semantic gains—especially customer topic recognition—but weak handling of payment deadlines and some numeric changes. I’ll keep that limitation explicit. Next I’ll test a multinomial contrast target that accounts for each question’s option count, then refit one selected readout using the 600 fitting-plus-development records.
```python
# Hypothesis: multinomial-tangent targets better balance boolean and multi-option questions than a common three-logit target scale.
note("Cue diagnosis: head inputs unchanged exactly under edit. Topic target probabilities improved account .229->.282, billing .369->.615, delivery .282->.679, refund .184->.331, technical .155->.531. Violations0->1 needs_review response changes original .804->.779 (wrong direction) to edited .330->.613 (right). Violations3 and invoice deadlines remain unreliable; no claim of complete numeric repair.")
# Two prespecified first-order inverse-softmax targets, independently solved on fit450.
tangent_specs={
    "Ksoft":centered(counts[:,None]*G),
    "Kmix":centered(counts[:,None]*(.5*G+.5*onehot))
}
tangent_results=[]
for tn,T in tangent_specs.items():
    yr=((T-L0c)[fit_rows]*RW[fit_rows,None])[mask[fit_rows]]
    pr=eigvecs.T@(Dt.T@torch.from_numpy(yr))
    for lam in [.003,.01,.03]:
        dw=(eigvecs@(pr/(eigvals+lam))).numpy()
        closed_form[(tn,lam)]=dw
        for strength in [.75,1.0]:
            pred=head_logits(w0+strength*dw)
            tangent_results.append((metric_logits(pred,dev_rows)[0],metric_logits(pred,dev_rows)[1],
                                    metric_logits(pred,fit_rows)[0],tn,lam,strength,np.linalg.norm(dw)))
print("Multinomial candidates: dev accuracy, Brier, fit accuracy, target, ridge, strength, norm")
for t in sorted(tangent_results,reverse=True): print(tuple(round(x,4) if isinstance(x,(float,np.floating)) else x for x in t))
# A single refit recipe will be frozen before any independent measurement.
print("Comparator selected earlier: mix3 ridge.003 strength.75, dev",
      metric_logits(head_logits(w0+.75*d_mix),dev_rows))
```
