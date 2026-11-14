# Lab 09 — Transaction sequences

## Question and setup

Does adding recent outgoing transactions improve detection beyond our existing
transaction features and four account-history counts?

For each target, selected up to eight earlier outgoing transactions from the same
bank/account within 24 hours, oldest first. Excluded all events at the target's
timestamp. Source-row order breaks ties between earlier events reproducibly; it
does not establish their actual within-minute chronology.

Each event has standardized log amount, one-hot currency and payment format, and
log-scaled age relative to the target: 23 inputs. Preprocessing was fitted on
training rows. A GRU produces a 16-value representation, concatenated with the
existing 44 inputs. The control supplies zeros for those extra 16 inputs.

Both models used the same training rows, initial classifier weights, batch order,
five epochs, seed 42, batches of 4,096, AdamW, and two CPU threads. No account-ID
embeddings were learned. Training: September 3–6; validation: September 7–8.

## Results

Validation: 965,524 transactions, 1,036 positives, 50 reviews per day.
Evaluated each model's final epoch.

| Measure | Existing features | With sequence |
|---|---:|---:|
| Average precision | 0.0864 | 0.2018 |
| Laundering caught / 100 reviews | 33 | 91 |
| False alerts / 100 reviews | 67 | 9 |
| Precision at budget | 33% | 91% |
| Recall at budget | 3.19% | 8.78% |
| Recorded five-epoch time | 15.8 s | 35.9 s |

The sequence model caught 46 cases on September 7 and 45 on September 8.
Both models selected only ACH transactions. Earlier boosting with history had
AP 0.1338 and 52 detections on the same validation set.

Timings include batch preparation, training, validation prediction, and AP each
epoch. They exclude initial history preparation and checkpoint saving. Both
runs construct history batches; the control skips GRU computation.

## Decision and limitations

Retain the sequence model as our strongest validation candidate so far and keep
boosting as a reference. The new branch improved broad ranking and the reviewed
cases. This does not isolate the benefit of order: the branch also adds historical
amounts, currencies, formats, timing, and model capacity.

91% precision means 91 positives among 100 reviews, not 91% of all laundering.
Recall is 91 / 1,036. One seed, a short synthetic-data window, and repeated use of
this validation set limit generalization claims. Final-test data remains reserved.

Verified every training/validation history's account, time window, order, and
padding; saved source-row alignment; validation labels; training-fitted encoder
categories and scaler statistics; sampled event features; and full predictions
reproduced from both checkpoints. Recomputed metrics match the saved results.

## Learning takeaways

- Sort and search once instead of repeatedly filtering millions of rows.
  `searchsorted(side="left")` supports strict exclusion at the target timestamp.
- Store history pointers as `int32` (71.6 MiB), store each event's features once
  (353.7 MiB), and expand sequences only for a batch. These are array sizes,
  not peak process memory. Memory mapping avoids loading entire cached inputs.
- Padding ID `-1` is a marker, not a safe array index: NumPy interprets it as the
  final row. Replace it before gathering, then zero padded inputs.
- Use the GRU output after the last real event. Empty histories get a zero
  representation. Changing padding values must not change predictions.
- Shuffle target examples while preserving each history's event order. Compute
  event age per target because the same historical event can serve many targets.
- Functions prepare batches; the model class owns learned parameters. IDs group
  events but are not model features. Unseen Reinvestment gets a zero format block.

Artifacts: `lab_09_training.csv`, `lab_09_model_comparison.csv`, prepared arrays
and scores in `data/processed/lab09/`, and `models/lab09_*.pt` checkpoints.
