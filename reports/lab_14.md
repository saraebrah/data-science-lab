# Lab 14 — Evaluate investigation notes

**Question.** Given the same alert and two retrieved passages, does each model produce an accurate observation and a useful, supported next check?

We chose three contrasting ACH alerts from the Lab 9 validation review queue and wrote reference questions before generating notes. Qwen and GPT-6 Luna each received the same evidence for each alert. A different follow-up could earn full credit if it would uncover new information.

| Alert | Model | Observation | Follow-up | Citation support |
|---|---|---|---|---|
| 695464 | Qwen | Counts and amount correct. Receiver time window could be clearer. | Useful but broad historical comparison. | P01 supports comparing payment volume. “Unusual payment patterns” is broader. |
| 695464 | Luna | Accurate. | Compare the payment amount and type with the sender’s history. | P06 supports both checks. |
| 3123427 | Qwen | Accurate. | Partly useful. The 24-hour count is known, and no comparison period or baseline is specified. | Partial. P03 concerns the timing of inflows and outflows. A high count alone cannot establish that sequence. |
| 3123427 | Luna | Accurate. | Compare the observed 35 payments with the sender’s usual volume. | P01 directly supports the comparison. |
| 3863638 | Qwen | Counts and amount correct. “No activity” could state the receiver’s 24-hour window explicitly. | Check typical payment size. Naming the sender would clarify whose history to inspect. | P06 supports the size check. |
| 3863638 | Luna | Accurate, with an explicit time window. | Compare the amount and ACH method with the sender’s usual activity. | P06 supports the check. |

**Decision.** Luna gave more specific follow-ups on these three cases. The small, deliberately chosen sample supports a qualitative comparison. Keep the six notes as a baseline for testing whether a lookup tool improves the investigation.

**Learning takeaways**

- Grade factual accuracy, usefulness, and citation support separately. A sensible question can cite a passage that supports only part of it.
- Look for the missing evidence. A count of 35 is observed. Whether 35 is unusual requires account history. Whether funds moved through quickly requires timestamps.
- On alert 695464, rewriting the description and including receiver history changed the top two passages from P07/P03 in Lab 13 to P01/P06. Compare models using the same retrieved context.
