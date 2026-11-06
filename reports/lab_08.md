# Lab 08 — Pre-training and fine-tuning

## Question and setup

Does masked-category pre-training help laundering detection with limited labels?

Pre-trained on 1,379,551 training-period transactions for three epochs, hiding
one categorical value per row and predicting it from the others. Laundering
labels were not used for this objective. Reconstruction heads predicted payment
currency, receiving currency, and payment format.

Compared scratch training with a pretrained encoder plus a fresh laundering
classifier. Both used the same architecture, initial classifier weights, batch
order, and 137,955 labeled rows containing 180 positives. This stratified 10%
sample simulates limited labels; real label availability may follow other patterns.
Both had five supervised epochs; all parameters remained trainable. Used seed
42, AdamW, batches of 4,096, and two CPU threads.

## Results

Validation: September 7–8, 2022; 1,036 positives; 50 reviews per day.
Evaluated the planned final epoch for each model.

| Measure | Scratch | Pretrained + fine-tuned |
|---|---:|---:|
| Average precision | 0.0227 | 0.0179 |
| Laundering caught / 100 reviews | 4 | 2 |
| Precision at budget | 4% | 2% |
| Recall at budget | 0.39% | 0.19% |
| Recorded total time | 7.5 s | 23.8 s |

Pre-training added 16.7 seconds. Supervised time includes training, validation
prediction, and AP calculation; pre-training time includes masking, training,
and reconstruction metrics. Input preparation and checkpoint saving are excluded.
The comparison matches labels and supervised updates, not total compute.

Final training reconstruction accuracy: payment currency 98.40%, receiving
currency 98.41%, payment format 43.17%. The currencies match in approximately
98.4% of training rows, making the visible currency an easy prediction baseline.
High reconstruction accuracy does not establish useful laundering features.

## Decision and checks

Pre-training hurt both AP and detections in this run. An easy reconstruction
objective may explain poor transfer, but we did not establish the cause. One seed,
label subset, and short training schedule do not settle pre-training's general value.

Keep boosting with history as the existing project reference. Its all-label
result is not a matched comparison for this experiment. Final-test data remained
untouched.

Verified subset size, positives, source-row IDs, time separation, validation
row/label alignment, matching model configurations, and full saved predictions
reproduced from both supervised checkpoints.

## Learning takeaways

- Pre-training and fine-tuning optimize different objectives: hidden-category
  cross-entropy first, binary laundering loss afterward. Evaluate the final task.
- Unknown ID 0 and MASK IDs have different roles. Unknown means unseen category;
  MASK means deliberately hidden. Known reconstruction targets start at zero.
- Transfer embedding and transformer weights; discard reconstruction heads.
  A fresh classifier learns the new target. Full fine-tuning also updates the encoder.
- `load_state_dict()` copies weights into existing modules by default; it does
  not make two models share parameter objects. A new optimizer starts fresh state.
- Memory mapping reads arrays without loading the whole file into RAM at once.
  Copy only selected numerical training rows; pre-training needs categories only.
- Separate seeded generators control shuffling and masking. Keep them across
  epochs so their random sequences advance rather than restart.
- AP summarizes ranking across recall levels. Precision here is detections / 100;
  recall is detections / 1,036. Reconstruction accuracy measures a different task.

Artifacts: `lab_08_pretraining.csv`, `lab_08_finetuning.csv`,
`lab_08_model_comparison.csv`; subset IDs and scores in `data/processed/lab08/`;
three checkpoints in `models/lab08_*.pt`.
