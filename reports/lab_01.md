# Lab 01 — Money-movement data audit

## Purpose

Understand the IBM AML transaction dataset and determine how to use it
for a chronological laundering-risk prediction experiment.

## Dataset

- File: HI-Small_Trans.csv
- Source: IBM Transactions for Anti Money Laundering, published on Kaggle.
- Synthetic data; labels indicate simulated money laundering.
- Transactions: 5,078,345.
- Recorded period: September 1–18, 2022.
- Payment types: ACH, wire, cheque, credit card, cash, reinvestment,
  and Bitcoin.

One row describes a recorded transaction between a sending and receiving
bank/account pair. It includes the timestamp, paid and received amounts,
currencies, payment format, and laundering label.

## Main findings

1. The first 1,000 rows are not representative of the full payment mix.
   Reinvestment accounts for 72% of the preview but 9.47% overall.

2. Rows are not globally ordered by timestamp. History calculations
   will require explicit temporal ordering.

3. Daily volume drops sharply after September 10:
   - September 10: 208,325 transactions.
   - September 11: 396 transactions.
   - September 18: 11 transactions.

   The recorded date range therefore does not represent 18 days of
   comparable activity. The reason for the drop is not yet established.

4. Payment mix around the coverage boundary:
   - Reinvestment accounts for 43.1% of September 1's transactions (481,056 records). No reinvestment transactions appear on subsequent dates in this dataset.
   - From September 11–18, all recorded transactions are ACH. The later
     records therefore have a different payment mix from the main activity
     window, although these counts do not explain why.

## Proposed prediction task

Score each newly recorded transaction for laundering risk using its
attributes and strictly earlier observable account activity.

This is an investigation-ranking task. It does not establish whether
a payment should be blocked before authorization.

## Information rules

- Identify an account endpoint using both bank and account identifiers.
- Exclude the laundering label and supplied laundering-pattern
  annotations from model inputs.
- Initially exclude features derived from historical laundering labels,
  because their availability at prediction time has not been established.
- Use only strictly earlier transactions for history features.
- Do not assume an ordering between transactions with identical timestamps.
- Keep monetary calculations separated by currency unless an explicit
  conversion method is available.
- Reserve final-test labels for final evaluation.

## Provisional time split

These dates are a proposal, not a finalized evaluation design:

- September 1–2: history-building context.
- September 3–6: training.
- September 7–8: validation.
- September 9–10: final test.
- September 11–18: retain separately while investigating sparse coverage.

Before adopting this split, check the Step 2 payment-mix results.
The short observation window limits conclusions about generalization.

## Validation performed

- [x] Daily payment counts sum to 5,078,345.
- [x] Counts combine correctly across chunks.
- [x] Summary dates are sorted.
- [x] Changing chunk boundaries does not change the summary.

Only mark checks that were actually run and passed.

## Limitations and open questions

- The data is synthetic; results will not establish real-world effectiveness.
- The observation window is too short for monthly seasonality or
  long-term drift analysis.
- Why does transaction volume collapse after September 10?
- Does early high volume reflect a different payment mix?
- How should we construct a manageable modeling subset while preserving
  relevant incoming and outgoing account histories?

## Next lab

Confirm the evaluation window, construct a manageable modeling dataset,
and build the first baseline.
