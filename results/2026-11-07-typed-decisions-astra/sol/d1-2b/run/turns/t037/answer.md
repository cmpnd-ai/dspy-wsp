Half strength gives a smaller 5.63-point gain, as predicted. I’ll test strength three next; the head reconstruction predicts deterioration beyond strength two.

```python
# Hypothesis: excessive amplification of the same dimensions reduces the gain, establishing an upper-dose limit.
save_edit(gn,g0*(1+torch.tensor(3*v3,dtype=torch.float32)))
note("Pass15 C strength.5 fit +.05625 SE .006039. Pass16 C strength3 same448 fit, upper-dose bracket; predicted below strength2.")
```
