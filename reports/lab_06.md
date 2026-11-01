# Lab 06 — Categorical embeddings

## Question and setup

Can learned category vectors improve the Lab 5 neural baseline?
Reused its numerical inputs, labels, and time splits. Replaced one-hot categories
with separate embeddings: payment currency (4 values), receiving currency (4),
and payment format (3). Combined these with 8 numerical features, giving 19 inputs
to the 32 → 16 → 1 classifier.

Used five epochs, batch size 4,096, AdamW with learning rate 0.001 and weight decay
0.0001, seed 42, unweighted loss, CPU with two threads. Parameter count changed
from 1,985 to 1,334, so this also changes model capacity.

## Results

Validation: September 7–8, 2022; 1,036 positives; 50 reviews per day.

| Model | AP | Caught / 100 reviews | Precision at budget | Recall at budget |
|---|---:|---:|---:|---:|
| Boosting with history | 0.1338 | 52 | 52% | 5.02% |
| One-hot neural network | 0.0821 | 35 | 35% | 3.38% |
| Embedding neural network | 0.0767 | 37 | 37% | 3.57% |

Embedding AP rose from 0.0296 after epoch one to 0.0767 after epoch five. Training
loss fell from 0.00807 to 0.00669. All 100 reviews were ACH. Five epochs, including
validation inference and AP calculation, took about 32 seconds.

Verified row/label alignment, category IDs against the fitted encoder, and full
validation predictions reproduced from the saved checkpoint. Final-test data
remained untouched.

## Decision and interpretation

Keep boosting with history as the reference. Embeddings caught two more cases
than the one-hot network but had lower AP. AP summarizes the pooled validation
ranking across thresholds; the review budget selects only 50 cases per day.
They can disagree. One seed and five epochs do not establish a reliable advantage.

All known payment vectors changed; the unknown vector stayed zero. Wire moved
the most (L2 change 0.940), despite having no positive training labels. Legitimate
examples also contribute gradients. Movement is not a risk or importance score.

## Learning takeaways

- Category IDs are lookup addresses. Embedding vectors are learned jointly with
  classifier weights; their coordinates have no predefined business meaning.
- `ModuleList` registers the tables so `model.parameters()` and checkpoints
  include them. Each categorical column has its own table.
- ID 0 uses a fixed zero vector via `padding_idx=0`. This fallback does not imply
  zero laundering risk; other inputs still affect the prediction.
- `mmap_mode="r"` accesses the old array on demand; we copy only its eight
  numerical columns. Numerical inputs plus integer IDs occupy about 125.3 MiB
  across both splits, versus 393.7 MiB for Lab 5 inputs. This is not peak RAM.
- `detach().clone()` makes an independent snapshot without gradient tracking.
  L2 change measures distance from that snapshot, not category importance.

Artifacts: `lab_06_training.csv`, `lab_06_model_comparison.csv`,
`lab_06_payment_embeddings.csv`, inputs/scores under `data/processed/lab06/`,
and `models/lab06_embedding_classifier.pt`.
