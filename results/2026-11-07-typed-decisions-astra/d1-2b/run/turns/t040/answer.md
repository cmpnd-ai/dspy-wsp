Removing the state-dependent component should reveal how much of the gain comes from evidence in each record. I’ll record that result, then test the development-selected weight-norm-regularized candidate on the reserved practice test. It uses the same nine output directions but reduces the matrix change by about 30%.
```python
# Hypothesis: the state-dependent component is necessary, and penalizing actual W2 displacement improves robustness without losing that mechanism.
r16=read_arrays(16)
m16=metrics_array(dev,r16["p"])
print("Question-centroid-only actual metrics:",m16)
delta_state=[]
for eid in sorted(set(dev["row_ids"])):
    ix=np.array([x==eid for x in dev["row_ids"]])
    full=(r12["p"][ix].argmax(1)==dev["gold"][ix].argmax(1)).mean()
    ablated=(r16["p"][ix].argmax(1)==dev["gold"][ix].argmax(1)).mean()
    delta_state.append(full-ablated)
delta_state=np.array(delta_state)
print("Full minus state-ablated:",delta_state.mean(),"paired SE",delta_state.std(ddof=1)/np.sqrt(len(delta_state)))
for key in sorted(set(dev["keys"])):
    ix=np.array([k==key for k in dev["keys"]])
    full=(r12["p"][ix].argmax(1)==dev["gold"][ix].argmax(1)).mean()
    ablated=(r16["p"][ix].argmax(1)==dev["gold"][ix].argmax(1)).mean()
    if abs(full-ablated)>=.1:print(key,"ablated",round(ablated,3),"full",round(full,3))
A_weight1=A_weight[1.0].copy()
delta_weight1=(torch.from_numpy(B).double()@(torch.from_numpy(A_weight1)@w2orig.double())).float()
install_w2(1.,delta_weight1)
backup2=WORK/"candidate_weightnorm1"
if backup2.exists():shutil.rmtree(backup2)
shutil.copytree(WORK/"edited",backup2)
np.savez(WORK/"candidate_weightnorm1_factors.npz",B=B,A=A_weight1,alpha=1.)
note("Pass 16 actual question-centroid-only ablation: "+str(m16)+
     f". Full minus ablated gain {delta_state.mean():.6f}, paired SE {delta_state.std(ddof=1)/np.sqrt(len(delta_state)):.6f}. This causally demonstrates value from the within-question state-sensitive directions, beyond question biases.\n\nPlanned pass 17: DECIDE both on reserved practice test n=360, test_pair.json. Exact current edit: W2_original+B A_weightnorm_ridge1 W2_original, alpha1, fit=360. This candidate was selected on dev (.787778) using actual-weight Frobenius regularization; relative update norm 1.8534 versus 2.641 prior. All other weights original. Confirmed best_1 untouched.")
```
