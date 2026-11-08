I’ll finish the complementary probe on development data, then restore the original norm gains. The candidate edit will go only into the last MLP’s output matrix and will be derived in closed form from the recovered write vectors.
```python
# Hypothesis: development-set head recomputation can predict a targeted W2 direction edit independently of its fitting examples.
note("Pass 9 completed: odd-gain fit_b probe, fwd_9.\n\nPlanned pass 10: READ edited dev, 180 IDs in probe_odd_dev.json. Exact edit remains post_ffn_norm.23 odd coordinates x1.5; even original. After this diagnostic pass the norm edit will be dropped. Proposed repair location remains layers.23.feed_forward.w2.weight only.")
print("Next:",str(WORK/"probe_odd_dev.json"),"edit:",edited())
```
