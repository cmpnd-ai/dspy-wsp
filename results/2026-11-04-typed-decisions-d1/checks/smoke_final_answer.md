**Final submission: original weights, with no edits retained.**

I tested a closed-form correction to `head.scorer.3.weight` targeting repeated option-selection biases, including “partial” trace outcomes and underestimated invoice urgency. Offline scorer reconstruction verified that the intervention changed the intended option margins.

- **Practice, 51 records:** accuracy improved by **6.27 percentage points ± 3.47 pp paired SE**.
- **Untouched validation, 50 records:** accuracy fell from **58% to 50%**, a change of **−8.00 pp ± 3.02 pp paired SE**.

The correction did not generalize, so I removed it. No weight change demonstrated a confirmed benefit.