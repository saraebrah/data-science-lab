# Lab 03 — Account-history features

## Objective

Test whether earlier incoming and outgoing account activity improves the Lab 2
transaction-level baseline at the same daily review capacity.

## Experiment

- Data: IBM AML `HI-Small_Trans.csv` (synthetic transactions).
- History: September 1–8, 2022; 4,214,445 transactions. September 1–2 supplies
  context before the first training example.
- Training: September 3–6; 1,379,551 transactions, with 1,800 positives.
- Validation: September 7–8; 965,524 transactions, with 1,036 positives.
- Historical features: incoming and outgoing counts for both the sending and
  receiving account, each identified by its bank/account pair.
- Window: [transaction timestamp - 24 hours, transaction timestamp). All events
  at the scored transaction's timestamp are excluded, including other records
  with that timestamp. Self-transfers count once in each direction.
- Earlier validation transactions can contribute observable activity to later
  validation features. Their laundering labels are excluded from the calculation.
- Both models use identical rows, the same original features, the same boosting
  settings, and a two-thread limit. The history model adds four numerical inputs.
- Review capacity: 50 transactions per day, or 100 across validation.

## Results

Source: `lab_03_history_metrics.csv`.

| Model | Average precision | Laundering caught | Precision at budget | Recall at budget | Fit and prediction |
|---|---:|---:|---:|---:|---:|
| Transaction-only | 0.066864 | 27 | 27% | 2.606% | 3.78 s |
| With account history | 0.133806 | 52 | 52% | 5.019% | 4.06 s |

Account history approximately doubled average precision. At the same 100 reviews,
the model caught 25 additional laundering transactions, raising precision by
25 percentage points. Both measures favor the history model in this validation
experiment. Runtime increased only slightly in this run.

The transaction-only run reproduced the Lab 2 baseline. The comparison supports
the usefulness of these four features as a group; it does not establish how much
each individual feature contributes.

## Verification and limitations

- Automated tests cover window boundaries, identical timestamps, bank/account
  identity, self-transfers, independence from future events, input-order
  invariance, and duplicate source-row rejection.
- Modeling tests check explicit feature selection, unchanged inputs, categorical
  identification after adding numerical columns, and unseen-category handling.
- All 11 project tests passed when preparing the comparison.
- Notebook checks verify complete, nonnegative history features and preserved
  transaction order after one-to-one joins.
- A direct-filtering notebook example confirmed the sender's incoming count.
- The final-test period, September 9–10, remains unevaluated.
- Results use synthetic data and two validation days, with repeated comparisons
  on that validation period. They do not establish performance on future periods.
- Selecting a completed day's top 50 scores is an offline capacity benchmark;
  immediate decisions would require an online selection policy.
- Counts measure activity, not transferred value or proof of illicit behavior.
  A zero means no matching activity in the supplied window, not a newly opened
  account. Complete historical coverage is a precondition of this interpretation.

## Decision

Retain the history model as the current reference. Inspect its remaining false
alerts and missed laundering transactions before choosing the next representation
or model change. Keep final-test labels reserved.

## Learning takeaways

- Temporal feature correctness depends on information availability. Sorting rows
  is insufficient: identical timestamps must not acquire an invented causal order.
- Cumulative counts at two strictly earlier cutoffs can calculate a trailing
  window efficiently. Subtracting counts before the window start preserves
  events exactly at the start while excluding events at the scored timestamp.
- `merge_asof` supports timestamp-based lookups; `allow_exact_matches=False`
  enforces a strict earlier-than boundary.
- Stable source-row identifiers support one-to-one joins and prediction
  traceability. Both cardinality and row order need verification.
- Arrow-backed strings can reduce identifier storage costs without discarding
  leading zeros. Measure memory rather than choosing dtypes by habit.
- Feature configuration and preprocessing must stay consistent. Adding numerical
  columns moves categorical positions in the transformed matrix; deriving those
  positions prevents silently applying the wrong feature semantics.
- A controlled feature comparison holds data, model settings, and evaluation
  fixed. Adding information can improve a model without changing its algorithm.
- Functions fit stateless feature construction and evaluation. The fitted pipeline
  appropriately retains learned encoder and model state.

### Compute and memory lessons

- We kept `with threadpool_limits(limits=2):` around model fitting and prediction
  for both experiments. This controls supported native CPU thread pools, restores
  previous settings afterward, and keeps concurrency settings consistent across
  the comparison. It does not impose a memory limit or control a GPU. See the
  [Lab 2 explanation](lab_02.md#cpu-threads-and-threadpool_limits).
- The history calculation and boosting experiments used CPU computation. GPU
  acceleration was not configured or benchmarked in these labs.
- `string[pyarrow]` stores identifier text in Arrow-backed buffers. This differs
  from pandas `category`, which stores a dictionary of categories plus row codes.
  Both are storage choices; their usefulness depends on the data. Arrow-backed
  strings preserved leading zeros in our account and bank identifiers.
- `history_parts.clear()` releases the list's references to intermediate batches
  after concatenation. It does not delete the combined history table. Other
  references can keep objects alive, and releasing Python objects does not
  guarantee an immediate drop in the process memory shown by macOS.
- Reading in chunks limits memory during each read. Collecting selected chunks,
  concatenating, sorting, and joining still allocates retained data and temporary
  objects. A final table's measured size is therefore smaller than the possible
  peak process memory required to construct it.
- Cumulative summaries and timestamp lookups avoid filtering the complete history
  separately for every target transaction. This changes the computational approach
  rather than simply using more CPU threads. Sorting and joins still require time
  and temporary memory.
- Saving `lab03_history_features.parquet` makes the feature table reusable across
  model experiments, avoiding repeated feature computation while trading disk
  space for saved preparation time. The source-row key preserves join alignment.
- The 3.78 s and 4.06 s measurements cover timed fitting and prediction blocks.
  They exclude history-feature construction. A complete runtime comparison should
  also account for preparation costs; small timing differences need repeated runs
  before being treated as reliable performance differences.

Source for thread-control behavior:
[threadpoolctl](https://github.com/joblib/threadpoolctl).
