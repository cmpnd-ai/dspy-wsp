**Submitted: a rank-one repair to `k.weight` only.** The validated weights are left in `edited/`. All backbone tensors and other head parameters remain unchanged.

### What changed and why
I added \(uv^\top\) to the key projection: \(u\) follows the head’s mean-query direction, while \(v\) reads state-dependent evidence from option activations. The direction was derived by closed-form ridge regression on 540 fitting examples—not gradient training or whole-tensor scaling.

The evidence supports a readout bottleneck:
- Ten dimensions carry about **90% of option activation energy**, but editing only those dimensions produced **no net development gain**.
- The complementary low-energy dimensions retained almost the entire repair’s benefit.
- Actual development accuracy at **0× / 0.5× / 1× / 1.5×** strength was **57.78% / 69.78% / 77.22% / 74.67%**, confirming a dose response and overshoot.
- Controlled cue changes showed stronger responses to delivery exceptions, payment deadlines, and constraint violations.

### Measured results

| Evaluation | Original | Edited | Paired gain ± SE |
|---|---:|---:|---:|
| Reserved practice, 180 cases | 59.11% | **74.78%** | **+15.67 ± 1.94 pp** |
| Hidden validation, 300 cases | 58.20% | **73.40%** | **+15.20 ± 1.40 pp** |
| All practice, including fitting cases | 57.91% | **77.87%** | **+19.96 ± 0.83 pp** |

On validation, **174 cases improved and 35 worsened**. The final pass reproduced the earlier validation result with the exact submission weights; it was a verification, not an independent replication.

I expect this to help because it strengthens an existing state-evidence signal rather than imposing fixed option preferences. Remaining limitations include fine-grained action distinctions and inconsistent allowlist sensitivity.