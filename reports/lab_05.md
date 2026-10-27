# Lab 05 — Neural baseline

## Question and setup

Can a small neural network improve on boosting using transaction and account
history features? Used the same training dates (September 3–6, 2022) and
validation dates (September 7–8), with 50 reviews per day. Final-test data
remained untouched. The source data is synthetic laundering data.

Prepared 44 float32 inputs: log1p and standard scaling for amounts/counts,
standard scaling for time features, and one-hot encoding for categories.
Fitted preprocessing on training only. Boosting uses different preprocessing,
so this compares complete modeling approaches, not architecture alone.

Trained a 44 → 32 → 16 → 1 network with ReLU hidden layers for five epochs:
AdamW, learning rate 0.001, weight decay 0.0001, batch size 4,096, seed 42,
unweighted binary cross-entropy, CPU with two threads.

## Results

| Model | AP | Caught / 100 reviews | Precision at budget | Recall at budget |
|---|---:|---:|---:|---:|
| Boosting, transaction only | 0.0669 | 27 | 27% | 2.61% |
| Boosting, with history | 0.1338 | 52 | 52% | 5.02% |
| Neural network, with history | 0.0821 | 35 | 35% | 3.38% |

Neural validation AP rose from 0.0346 to 0.0821; training loss fell from 0.00814
to 0.00660. All 100 neural reviews were ACH. Five epochs, including validation
inference and AP calculation each epoch, took about 28 seconds. Input preparation
is excluded; this is not a controlled speed comparison with boosting.

## Decision

Keep boosting with history as the reference. This neural baseline ranks cases
usefully but trails that reference on both AP and detections at the review budget.
AP was still rising at epoch five; whether more training helps is untested.
This experiment does not establish that neural networks are generally worse.

## Learning takeaways

- `log1p` compresses large amounts/counts; scaling helps optimization. One-hot
  encoding avoids treating category storage codes as ordered numbers.
- Float32 uses half the array memory of float64. Training inputs occupy 231.6
  MiB; validation inputs occupy 162.1 MiB. These exclude other process memory.
- `torch.from_numpy` shares array memory. Batching limits intermediate
  calculations; all input arrays still remain in RAM.
- Two PyTorch threads limit parallel CPU calculations. `num_workers=0` avoids
  extra data-loader processes; it is a separate setting.
- `BCEWithLogitsLoss` takes raw logits and handles sigmoid internally. Use
  sigmoid for prediction scores. The initial output bias uses training log-odds
  to start predictions near the rare positive rate.
- Each batch clears old gradients, calculates loss, backpropagates, then updates
  weights. Training loss and validation AP answer different questions; lower
  loss does not guarantee a better ranking.
- The model class owns learned weights. Prediction is a function. `eval()` sets
  evaluation behavior; `inference_mode()` disables gradient tracking for scoring.
- Saved preprocessing, weights, and source-row order have different roles.
  Verify score/label row alignment before evaluation. A checkpoint alone does
  not contain the fitted preprocessing.

Artifacts: `lab_05_training.csv`, `lab_05_model_comparison.csv`, prepared inputs
and scores under `data/processed/lab05/`, and `models/lab05_neural_baseline.pt`.
