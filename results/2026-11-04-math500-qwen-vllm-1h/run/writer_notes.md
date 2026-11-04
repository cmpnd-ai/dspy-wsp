Original diagnostic: 8/8 sampled items cut at 2048; repeated self-questioning and no boxed answer. Pass 3/4 block-23 channel 9,46 half/zero probes had paired gain 0 and all remained cut; that edit was removed.
Teacher-forced two transcripts: MLP activity near 'Wait' broad, but block 3 channel 382 had the largest consistently enhanced per-channel mean abs values (0.627 and 0.482 difference near vs far), motivating a single-channel causal probe. Correlational only.
Current edit: zero block-3 mlp.down_proj column 382 only. Pass 6 planned paired against original on train_5238/train_912/train_204/train_6074, max_tokens=2048.

Pass 7 planned: exact current edit is zeroed column 382 of model.language_model.layers.3.mlp.down_proj.weight (1,024 entries; other entries unchanged). Paired original/edit generation, max_tokens=2048, on fresh IDs: train_0, train_1, train_2, train_3, train_4, train_5, train_6, train_7, train_8, train_9, train_10, train_11, train_12, train_13, train_14, train_15, train_16, train_17, train_18, train_19, train_20, train_21, train_22, train_23.

Pass 8 planned: retain the zeroed block-3 MLP down_proj column 382 edit. Paired original/edit generation on fresh IDs train_24, train_25, train_26, train_27, train_28, train_29, train_30, train_31 with max_tokens=32768 (evaluation setting).

Final decision: removed the zeroed block-3 MLP down_proj column 382 edit. It showed +1/24 on pass 7 at 2,048 tokens but remained cut on all 24; on pass 8 at the full 32,768-token budget it scored 1/8 vs original 2/8 (paired gain -0.125, SE 0.227) and cutoffs increased from 6/8 to 7/8. The positive signal was weak and did not generalize to the evaluation-length test, so submit original weights.
