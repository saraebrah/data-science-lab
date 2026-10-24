# Lab 04 — Error analysis

## Question

Does removing payment format help detect non-ACH laundering without reducing
overall performance?

## What we inspected

- Selected five highest-scored false alerts, five random misses, and five random
  caught cases. These examples generate hypotheses; they do not establish rates.
- A separate full-table check confirmed that all 100 original validation reviews
  were ACH: 52 caught laundering and 48 false alerts.
- ACH represents 12.7% of training transactions but 87.9% of training positives.
  It dominates positive examples, rather than total transaction volume.

## Experiment and results

Retrained with `payment_format` excluded from both feature extraction and
preprocessing. Kept all other features, training settings, rows, and the daily
budget fixed. Training: September 3–6, 2022; validation: September 7–8.
Both models received 100 reviews total, or 50 per day.

| Measure | With format | Without format |
|---|---:|---:|
| Average precision | 0.1338 | 0.0301 |
| Laundering caught | 52 | 19 |
| Precision at budget | 52% | 19% |
| Non-ACH reviews | 0 | 54 |
| Non-ACH laundering caught | 0 | 0 |

The 54 non-ACH reviews comprised 5 cash, 29 cheque, 18 credit-card, and 2 wire
transactions. All were negative examples. The remaining 46 reviews were ACH,
catching 19 positives.

## Decision

Keep payment format in the reference model. Removing it broadened the payment
mix of reviews but reduced detection performance in this experiment. This
supports its usefulness alongside the current features and model settings.
It does not explain why non-ACH detection is weak or prove that scarce positive
examples caused the misses. Final-test labels remain reserved.

## Learning takeaways

- Compare amounts within the same currency. A large Yen number cannot be
  compared directly with a smaller Euro number.
- Reviews are not all errors: selected positives are detections, while selected
  negatives are false alerts. Missed positives are outside the review budget.
- Feature ablation removes an input and retrains under fixed conditions. Keep
  reporting metadata separate from the actual model inputs.
- More varied reviews do not necessarily catch more laundering. Evaluate both
  overall results and results by payment type.
- Source-row identifiers let us retrieve specific CSV records in one chunked
  scan. Assign positions before filtering, preserve the original source file,
  and validate joins.
- Saving the review table as Parquet lets us revisit scores and selections
  without retraining. Small case samples support investigation; full-table checks
  and controlled experiments support broader conclusions.

## Next

Compare a small neural-network baseline with boosting using the same data and
evaluation rules.
