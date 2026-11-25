# Lab 12 — Retrieve guidance for a transaction alert

**Question.** Can a reviewer find a useful follow-up passage among the first three search results?

We sampled five ACH alerts from Lab 9’s validation review queue. Each query gave the payment amount, currency, and the sender’s incoming and outgoing counts over the preceding 24 hours. We searched 12 teaching paraphrases of [FINTRAC’s guidance for financial entities](https://fintrac-canafe.canada.ca/guidance-directives/transaction-operation/indicators-indicateurs/fin_mltf-eng).

Before ranking, the reviewer selected P03 for all five alerts. P03 suggests checking the timing of funds entering and leaving an account. The counts give a reason to inspect timestamps. Establishing whether incoming funds were forwarded requires more evidence.

| Search method | First three results for every alert | P03 first | P03 in top three |
|---|---|---:|---:|
| TF-IDF (shared words and phrases) | P06, P03, P01 | 0/5 | 5/5 |
| Original text encoder (vector similarity) | P07, P03, P05 | 0/5 | 5/5 |

**Decision.** Showing two passages would have put P03 in front of the reviewer for every sampled alert. The methods tied on that outcome. All five cases had the same relevant passage, so a fixed list starting with P03 would also succeed. A choice between search methods for broader use needs alerts with different review questions.

**Learning takeaways**

- Fit TF-IDF on the passages, then transform the alerts using that vocabulary. The five descriptions produced identical TF-IDF vectors even though their amounts and counts differed.
- A high similarity score can surface a plausible hypothesis. The encoder ranked P07 first, a passage about forwarding money to beneficiaries. A reviewer would need transaction-level evidence to establish that link.
- Record relevance judgments before looking at rankings. Here relevance meant “helps decide what to check next.” A transaction’s laundering label answers a different question.

All ten saved rankings were reproduced from the alert texts, passages, and original encoder. See [the per-alert comparison](lab_12_retrieval_comparison.csv).
