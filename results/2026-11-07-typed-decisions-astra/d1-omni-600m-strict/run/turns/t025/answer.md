The protected direction still separates near-due from later-due invoices on the second discovery half, while unrelated option contrasts are much smaller. I’ll route it through two scorer neurons whose output weights are almost zero. Setting their inputs to opposite signs and their outputs to opposite weights gives a linear readout of this one direction, using GELU’s identity \(G(x)-G(-x)=x\).
This changes only two rows of the scorer’s first linear layer, their biases, and two output weights. I’ll start with gain 2 and test the actual model—not a simulated scorer—for both intended effects and collateral changes.
I’ll save the two-neuron deadline readout, with all encoder and transformer-head weights restored to original, and log the exact edit before testing it.
The code below installs the gain-2 deadline readout in the two nearly unused scorer neurons and prepares a read pass on discovery half B.
```python
# Hypothesis: explicitly reading the protected payment-deadline direction will raise overdue urgency without broadly changing unrelated rankings.
SC1="head.scorer.1.weight"
SCB="head.scorer.1.bias"
SC3="head.scorer.3.weight"
orig_sc1=load_original(SC1)
orig_scb=load_original(SCB)
orig_sc3=load_original(SC3)
ln_gain=load_original("head.scorer.0.weight").numpy()
ln_bias=load_original("head.scorer.0.bias").numpy()
cue_neurons=(594,746)
d_payment=protected_directions[32].astype(np.float32)
def install_linear_cues(cues):
    # Each cue is (neuron_pair, unit direction in centered option-feature space, gain).
    # This is an algebraic weight construction, not a forward evaluation or optimization.
    clear_edits()
    a,b,c=orig_sc1.clone(),orig_scb.clone(),orig_sc3.clone()
    for (j,k),direction,gain in cues:
        row=torch.from_numpy((np.asarray(direction,dtype=np.float32)/ln_gain).copy())
        offset=-float(np.asarray(row)@ln_bias)
        a[j]=row; a[k]=-row
        b[j]=offset; b[k]=-offset
        c[0,j]=gain; c[0,k]=-gain
    save_edit(SC1,a);save_edit(SCB,b);save_edit(SC3,c)
payment_gain=2.0
install_linear_cues([(cue_neurons,d_payment,payment_gain)])
np.save(WORK/"payment_direction.npy",d_payment)
payment_read_req={"type":"read","model":"edited","from":{"pass":1,"ids":read_ids_b},"layers":[16]}
(WORK/"payment_read_b.json").write_text(json.dumps(payment_read_req))
note("## Pass 7 planned: causal payment readout, gain 2\nExact edit: scorer neurons 594,746 (original output weights -3.41e-6,+3.75e-6) receive +/- protected payment direction divided coordinatewise by scorer LN gain; biases cancel LN offset; output weights +/-2. GELU antisymmetry creates a linear directional readout. Direction derived from 16 matched discovery-A invoice counterfactuals, projected off 53 fixed option/nuisance means and 32 unrelated-contrast PCs from discovery A. Other tensors original. "+json.dumps(edit_manifest())+
     "\nRead discovery B (200 cases/1000 questions, payment_read_b.json), compare original pass 3. Predict increased urgency for near/overdue invoices with limited unrelated changes. This is discovery/tuning, not final confirmation.")
print("Edit manifest:",edit_manifest())
```
