# Lab 11 — Fine-tuning text embeddings

## Question and setup

Can adapting Lab 10's pretrained encoder to banking categories improve retrieval?

Reused the 1,270-question corpus as both the training pool and searchable
collection. Constructed one triplet per question: anchor, a different question
from the same category, and a question from another category. All three roles
came from the training pool. The 318 Lab 10 queries served only as development
queries; 400 questions from the official BANKING77 test split supplied final
evaluation across the same ten categories.

Fully fine-tuned `all-MiniLM-L6-v2` for two epochs using normalized embeddings
and loss `max(0, 0.2 + similarity(anchor, negative) - similarity(anchor, positive))`.
Used seed 42, batches of eight triplets (24 texts), AdamW with learning rate
0.00002 and weight decay 0.01, CPU, and two threads. Compared the saved original
encoder with the planned epoch-2 encoder. No test-driven tuning was performed.

## Results

| Final-test measure | Original | Fine-tuned |
|---|---:|---:|
| Correct first results | 362 / 400 | 365 / 400 |
| Top-1 accuracy | 90.50% | 91.25% |
| Queries with a relevant result in top five | 387 / 400 | 385 / 400 |
| Hit@5 | 96.75% | 96.25% |
| MRR@5 | 0.9289 | 0.9328 |

Top-1 improved by 0.75 percentage points: 13 queries gained a correct first
result and ten lost one. Hit@5 fell by 0.50 points; MRR@5 rose slightly.
Higher MRR can coexist with lower Hit@5 when better placement on some queries
outweighs losses on others.

Development results followed the same mixed direction: top-1 rose from 90.88%
to 92.45%, Hit@5 fell from 96.86% to 95.91%, and MRR@5 rose from 0.9347 to 0.9382.
Epoch training loss fell from 0.0360 to 0.0127; active triplets fell from 24.33%
to 9.45%. Training took 111.5 seconds, excluding evaluation and model saving.

## Decision and limitations

The learning experiment is complete. Fine-tuning improved the training objective
but did not establish clear overall retrieval superiority. Keep the original
encoder as the general reference and preserve the adapted encoder for comparison.
Neither result establishes suitability for laundering investigations.

Category agreement is an imperfect relevance label; some sampled positives were
semantically weak. This was one seed, one fixed triplet sample, and a known-corpus
retrieval task. Small test differences do not establish a reliable advantage.
Normalized text overlap and conflicting duplicates were checked; all 400 selected
test questions remained. Near-duplicates and original pretraining exposure were
not ruled out. The AML transaction final test remains untouched.

Verified training-only triplet membership and relationships, query separation,
source text/labels, saved row IDs, both full test-score matrices reproduced from
saved encoders, and independently recomputed metrics.

## Learning takeaways

- Triplet training teaches relative similarity; a margin is not a probability.
  Labels define what similarity the model learns, including their imperfections.
- `encode()` performs inference. Training uses `preprocess()` and a direct
  forward pass so gradients reach the encoder; `model.train()` restores training
  mode after evaluation.
- Lower loss and fewer margin violations do not guarantee better retrieval.
- Re-encode both queries and corpus after changing encoder weights. Their vectors
  must come from the same embedding space.
- Small batches limit activation memory. Training additionally needs gradients
  and optimizer state, so its memory requirements exceed inference.
- `save_pretrained()` preserves a loadable encoder, configuration, and tokenizer.
  Saving both versions makes comparisons reproducible.

Artifacts: `lab_11_development.csv`, `lab_11_test_comparison.csv`, prepared data,
scores and config under `data/processed/lab11/`, and the original/fine-tuned
encoder directories under `models/lab11_*_encoder/`.
