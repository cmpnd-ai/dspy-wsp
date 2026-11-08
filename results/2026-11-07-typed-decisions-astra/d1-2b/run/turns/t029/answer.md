I’ll compare the actual half-strength logits with the head-only prediction, then test unit strength. Agreement would show that the edit acts through the intended last-write mechanism rather than an unmodeled backbone effect.
```python
# Hypothesis: actual option-logit changes match the norm-aware final-write prediction, establishing the intervention's causal mechanism.
r11=read_arrays(11)
pred_l,pred_p=dose_predictions[0.5]
real_l=r11["l"]
print("Actual half-strength metrics:",metrics_array(dev,r11["p"]))
print("Prediction error mean/max logits:",np.mean(np.abs((real_l-pred_l)[dev["mask"]])),np.max(np.abs((real_l-pred_l)[dev["mask"]])))
print("Prediction error mean/max probabilities:",np.mean(np.abs(r11["p"]-pred_p)),np.max(np.abs(r11["p"]-pred_p)))
print("Argmax agreement",np.mean(r11["p"].argmax(1)==pred_p.argmax(1)))
for lam in [1.,10.,30.,100.]:
    dwt=(torch.from_numpy(B).double()@(torch.from_numpy(A_by_lambda[lam]).double()@w2orig.double()))
    print("ridge",lam,"relative delta norm",float(dwt.norm()/w2orig.norm()))
del dwt
install_w2(1.0)
note("Pass 11 actual half-strength READ dev: "+str(metrics_array(dev,r11["p"]))+
     f"; predicted-vs-actual logit MAE {np.mean(np.abs((real_l-pred_l)[dev['mask']])):.6g}, argmax agreement {np.mean(r11['p'].argmax(1)==pred_p.argmax(1)):.6f}."+
     "\n\nPlanned pass 12: READ edited dev, same candidate_dev.json and same 180 records. Exact edit W2_original + 1.0*delta_w2 (ridge=1, rank<=9); norms and all other tensors original. Unit-strength predicted metrics: "+str(metrics_array(dev,dose_predictions[1.0][1]))+".")
```
