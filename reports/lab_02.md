# Lab 02 — Transaction-level baseline

## Objective

Determine whether transaction amounts, currencies, and time improve
laundering-risk ranking beyond a payment-type-only comparison.

## Data and experiment

- Dataset: IBM AML `HI-Small_Trans.csv`, synthetic money-movement records.
- Training: September 3–6, 2022; 1,379,551 transactions, including 1,800
  laundering transactions.
- Validation: September 7–8, 2022; 965,524 transactions, including 1,036
  laundering transactions.
- All transactions in these periods were retained. No class resampling was
  needed after selecting columns and using compact data types.
- Prepared DataFrame memory: approximately 47.4 MiB for training and 33.1 MiB
  for validation. These measurements exclude training intermediates and other
  process memory.
- Final-test dates, September 9–10, were not evaluated in this lab. The sparse
  ACH-only records after September 10 remain outside this initial experiment.

Both splits contain legitimate and laundering transactions. ACH accounts for
most positive examples. Wire has no positive examples in either split, making
wire laundering recall undefined. Bitcoin has 12 validation positives: one
additional detection changes its recall by 8.3 percentage points.

The first model uses amount paid, amount received, hour, weekday, payment
currency, receiving currency, and payment format. It excludes the target and
source-row identifier. Account history is reserved for a later experiment.

The payment-type comparison assigns each validation transaction its payment
type's training laundering rate, with the overall training rate as a fallback.
The boosting pipeline fits an OrdinalEncoder on training data, marks its three
encoded columns as categorical, and uses 50 boosting iterations, 15 maximum
leaves per tree, a 0.05 learning rate, and L2 regularization of 1.0. Automatic
early stopping is disabled; the seed is 42 and CPU threads are limited to two.
The notebook recorded 3.7 seconds for fit and prediction on this run.
Installed versions at completion: pandas 3.0.6, NumPy 2.5.3, and scikit-learn
1.9.1.

## Results

Source: `lab_02_baseline_metrics.csv`. All results below use validation data.

| Ranking | Average precision | Reviews | Laundering caught | Precision at budget | Recall at budget |
|---|---:|---:|---:|---:|---:|
| Payment type | 0.006842 | 100 | 2 | 2.00% | 0.193% |
| Gradient boosting | 0.066864 | 100 | 27 | 27.00% | 2.606% |

Both measures favor boosting. Average precision increased approximately 9.8
times, while detections under the same review budget increased from 2 to 27.
Transaction-level features therefore provide useful ranking information beyond
payment type in this validation experiment.

The review budget is 50 transactions per day, giving 100 reviews across two
days. Boosting's selected reviews include 27 positives and 73 negatives. Its
recall is 27 / 1,036 = 2.606%. With only 100 reviews, even selecting 100 positives
would give recall of at most 100 / 1,036 = 9.653%.

## Checks and limitations

- Prepared row and positive counts match the Step 1 audit.
- Notebook assertions verify sorted timestamps, unique source rows, disjoint
  splits, and chronological separation.
- A controlled two-day example verifies that the evaluation applies a separate
  daily budget and calculates precision and recall correctly.
- Metric arithmetic agrees with the saved detection and review counts.
- Evaluation selects a completed day's highest scores. An online system would
  require a policy for making decisions as transactions arrive.
- Ties are broken randomly with seed 42, independently of labels. Budget results
  for the payment-type comparison can vary with the tie-breaking seed.
- Results concern synthetic data and a two-day validation window. They do not
  establish real-world effectiveness or long-term stability.
- Training assumes its target labels are available when fitting. Label discovery
  times are not modeled.
- Model outputs have not been checked for calibration; ranking improvement does
  not establish that the probabilities are accurate.

## Decisions and next experiment

Keep this configuration and these validation results as the reference baseline.
Retain the original class proportions and hold the final test for later
evaluation. The next experiment will test whether earlier incoming and outgoing
account activity adds useful signal, using the same ranking metrics and budget.

## Learning takeaways

- Choose data types deliberately to reduce memory use. A validated binary
  `is_laundering` label needs only `int8` (one byte per value), rather than
  `int64` (eight bytes per value). Check that values are 0 or 1 before casting;
  converting a data type does not validate the label's meaning.
- Repeated values such as payment formats and currencies can use pandas
  `category`. Pandas stores a category dictionary plus integer codes for each
  row, which can be more compact than storing repeated strings. Measure actual
  memory usage; categorical conversion is not beneficial for every column,
  especially when nearly every value is distinct.
- Categorical storage preserves the original labels. There is no need to
  reverse the conversion before training. Model preprocessing must handle
  categories consistently: fit the encoding on training data and apply it to
  validation and test data. Pandas' independently assigned `.cat.codes` are
  not a shared model encoding and do not imply a numerical order or distance.
- A fitted pipeline retains preprocessing and model state together. Explicitly
  selecting input features helps prevent accidental target or metadata leakage.
- Encoded categories must be declared categorical for this boosting model;
  integer encoding alone does not establish meaningful numerical distances.
- Average precision summarizes ranking quality across score thresholds, while
  precision and recall at a daily budget describe a particular review capacity.
  Improvement factors can differ while both measures favor the same model.
- Recall depends on both selection quality and review capacity. Better ranking
  helps allocate reviews, but a limited budget bounds how many positives can be
  investigated.
- Functions suit stateless feature construction and evaluation. Existing encoder,
  estimator, and pipeline classes appropriately retain learned state.
