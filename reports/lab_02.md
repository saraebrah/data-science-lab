# Lab 02 — Transaction-level baseline

Report in progress. Add model results and final decisions at the end of the lab.

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

