The first correction improves the reserved practice cases by 6.36 percentage points, but the stronger regularization settings barely move rankings. I’ll examine lower regularization and intervention strengths on the fit cases only, then test one chosen direction causally.

```python
# Hypothesis: useful scorer features have low variance and require a less strongly damped directional edit.
for ridge in [.01,.1,1,3]:
    delta=torch.linalg.solve(gram+ridge*torch.eye(1024,dtype=torch.float64),rhs).float().numpy()
    deltas[ridge]=delta
    print("ridge",ridge,"norm",round(float(np.linalg.norm(delta)),2),
          [(a,report(a*delta,fit)) for a in [.25,.5,1,1.5]])
with notes.open("a") as f:
    f.write("\nScorer reconstruction max error 1.91e-6. Closed-form centered GELU-feature residual regression, 450 fit cases only, target calibrated log gold (floor .015). Ridge 10: fit +.0604±.0087, reserved practice +.0636±.0094; ridge100 +.0284±.0067 test, ridge1000 +.0067±.0038. No saved edit yet. Exploring lower regularization using fit cases only before causal dose tests.\n")
```
