The question direction cleanly separates security severity from every other question in both discovery halves. I’ll use it to gate the security readouts: paired GELU neurons should pass the cue correction on severity questions and remain nearly inactive elsewhere. The next pass will test that prediction directly.
```python
# Hypothesis: a security-severity gate will preserve cue-driven severity corrections while eliminating unrelated ranking changes.
def install_gated_security(gains, gate_scale=12.0, gate_threshold=9.5):
    install_linear_cues(base_cues)
    a=load(SC1).clone()
    b=load(SCB).clone()
    c=load(SC3).clone()
    for (j,k),direction,gain in zip(sec_specific_pairs,sec_specific_directions,gains):
        # In the gate's positive region, GELU(g+x)-GELU(g-x) is approximately 2x;
        # in its negative region both neurons are inactive. This is weight construction only.
        row_plus=(gate_scale*d_security_gate+direction)/np.asarray(ln_gain)
        row_minus=(gate_scale*d_security_gate-direction)/np.asarray(ln_gain)
        a[j]=torch.from_numpy(row_plus.astype(np.float32))
        a[k]=torch.from_numpy(row_minus.astype(np.float32))
        b[j]=-float(row_plus@np.asarray(ln_bias))-gate_threshold
        b[k]=-float(row_minus@np.asarray(ln_bias))-gate_threshold
        c[0,j]=gain/2
        c[0,k]=-gain/2
    save_edit(SC1,a)
    save_edit(SCB,b)
    save_edit(SC3,c)
install_gated_security(sec_specific_gains_moderate)
note("\n## Pass 18 planned: gate-localized security readouts, moderate dose\nExact edit: base payment4+violation2 plus same four security cue directions/gains3,3,2,6 as pass17. Security pairs now rows (12*question_gate +/- cue)/scorerLN_gain, biases cancel LN offset then subtract9.5, outputs +/-gain/2. Gate derived from discovery-A question means protected against option/content/cue variation. B gate projections security severity [.0352,.0635], others[-.0213,.0146], clear separation. "+json.dumps(edit_manifest())+
     "\nRead discovery B 200 cases/1000 questions via payment_read_b.json. Predict preserved severity effects but disappearance of unrelated changes versus validated base repair pass10. Gating has not yet been accepted causally.")
print(edit_manifest())
```
This prepares the gated edit. Its predicted specificity still needs to be tested with a forward pass.