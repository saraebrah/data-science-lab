# Lab 07 — Tabular transformer

## Question and setup

Does adding a transformer block help an embedding classifier on our money-movement
case study? Each categorical column becomes one 8-value token. Attention mixes
the three features within a transaction; it does not mix separate transactions.

Compared an embedding-only control with a model adding one transformer block:
two attention heads, layer normalization, residual connections, and an 8 → 32 → 8
feed-forward network. Flattened tokens supply 24 values; eight numerical features
bring the classifier input to 32. The classifier uses 32 → 16 → 1 layers.

Both runs used the same shared initial weights, batch order, data splits, five
epochs, batch size 4,096, AdamW (learning rate 0.001, weight decay 0.0001), seed 42,
and two CPU threads. Dropout was zero. Normalization precedes each block operation.

## Results

Validation: September 7–8, 2022; 1,036 positives; 50 reviews per day.

| Measure | Embedding control | Transformer |
|---|---:|---:|
| Average precision | 0.0598 | 0.0616 |
| Laundering caught / 100 reviews | 29 | 32 |
| Precision at budget | 29% | 32% |
| Recall at budget | 2.80% | 3.09% |
| Five-epoch run time | 31.9 s | 43.4 s |
| Parameters | 1,913 | 2,785 |

Both selected only ACH transactions. Timings include training, validation
prediction, and AP calculation each epoch; they exclude input preparation.
Verified row/label alignment, category-ID mapping, and full validation predictions
reproduced from both saved checkpoints. Final-test data remained untouched.

## Decision

The transformer improved both measures over its control in this run, with more
parameters and runtime. This compares the whole transformer block, not attention
alone. One seed and five epochs do not establish a reliable general advantage;
both validation AP curves were still rising.

Keep boosting with history as the reference: AP 0.1338 and 52 detections. The
Lab 6 embedding model also scored higher (AP 0.0767, 37 detections). Lab 7's
control uses widths 8/8/8 rather than Lab 6's 4/4/3, changing capacity and
initialization. Larger embeddings did not improve this run.

## Learning takeaways

- A category's lookup vector is shared across transactions. Attention makes its
  output depend on the other categorical features in that transaction.
- A token has a common width; here two heads split width 8 into widths of 4.
  Attention rows sum to one in the Step 1 demonstration with dropout disabled.
- Residual connections add learned updates to existing representations. Layer
  normalization works across each token's coordinates; the feed-forward network
  transforms each token separately.
- Matching seeds alone is insufficient across different architectures. Shared
  modules were initialized first with matching shapes; each run also reset its
  data-loader generator.
- A shared `train_one_epoch()` function keeps training consistent. Weight batch
  losses by batch size so the shorter final batch is counted correctly.
- Attention weights describe internal mixing, not causal feature importance.
  Compare prediction quality and compute cost rather than model complexity alone.

Artifacts: `lab_07_training.csv`, `lab_07_model_comparison.csv`, scores under
`data/processed/lab07/`, and both `models/lab07_*.pt` checkpoints.
