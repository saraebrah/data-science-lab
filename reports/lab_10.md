# Lab 10 — Text retrieval

## Question and setup

Does a pretrained text encoder retrieve related payment questions better than
word overlap?

Used ten payment/transfer categories from BANKING77's published training file.
After excluding conflicting normalized duplicates and deduplicating text,
split 1,588 questions into a 1,270-question corpus and 318 evaluation queries,
stratified by category with seed 42. The official test split was not used.
These are banking questions, not investigation narratives linked to our AML data.

Compared corpus-fitted TF-IDF with word unigrams/bigrams against the unchanged
`sentence-transformers/all-MiniLM-L6-v2` encoder. Encoded text on CPU with two
threads, batches of 32, and normalized 384-value vectors. Only text entered the
retrievers; category matches defined relevance for evaluation. Ties retained
corpus order. No fine-tuning or vector database was needed.

## Results

| Measure | TF-IDF | Embeddings |
|---|---:|---:|
| Same-category first result | 237 / 318 | 289 / 318 |
| Top-1 accuracy | 74.53% | 90.88% |
| At least one same-category result in top five | 303 / 318 | 308 / 318 |
| Hit@5 | 95.28% | 96.86% |
| MRR@5 | 0.8288 | 0.9347 |

Embeddings gained 16.35 percentage points at rank 1, but only 1.57 points on
Hit@5. Their main advantage was putting a relevant question first. They alone
were correct at rank 1 for 60 queries; TF-IDF alone for eight. Both were correct
for 229 and both missed for 21.

For a transfer missing from a balance, embeddings retrieved the balance issue;
TF-IDF selected a pending-transfer question. For an unspecified pending payment,
TF-IDF retrieved a card-payment question and embeddings a transfer question.
The latter query is ambiguous: its assigned card category does not make every
transfer interpretation unreasonable.

## Decision and checks

Keep pretrained embeddings as the leading retrieval baseline and retain TF-IDF
for comparison. Category agreement is a proxy for relevance, not proof of useful
investigation assistance. This is one random split of ten categories; paraphrases
can remain across splits, and pretrained-model exposure was not audited.

Verified source text/labels and saved row IDs, disjoint normalized text across
splits, corpus-only TF-IDF fitting, unit-length embeddings, similarity matrices,
and all per-query and aggregate metrics. Verification reused saved vectors;
it did not rerun the encoder.

## Learning takeaways

- Tokens are vocabulary pieces, not necessarily whole words. A text encoder
  computes a representation from context; a category embedding is a lookup.
- `encode()` performs inference here: model weights stay unchanged.
- For unit-length vectors, matrix multiplication computes cosine similarities.
  Similarity is not a relevance probability or a calibrated cross-method score.
- Hit@5 asks whether any relevant result appears. MRR@5 averages the reciprocal
  rank of the first relevant result, assigning zero for no hit in five.
- Keep query/corpus ordering aligned with score rows/columns and save their IDs.
- The float32 embeddings occupy 2.33 MiB combined; this excludes model weights
  and working memory. Batching limits temporary inference allocations.

Artifacts: `lab_10_retrieval_comparison.csv`, `lab_10_query_results.csv`, and
cached vectors, scores, IDs, and vectorizer under `data/processed/lab10/`.

Sources: [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets#banking),
[model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
