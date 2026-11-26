# Lab 13 — Draft and check a transaction review note

**Question.** Can a model turn retrieved guidance into a useful next check for an investigator?

In the [lab notebook](../notebooks/13_grounded_review_notes.ipynb), we gave local Qwen `qwen3:4b` and OpenAI `gpt-4.1-mini` the same prompt, Lab 9 alert, two retrieved passages (P07 and P03), and two-field output schema. This is retrieval-augmented generation (RAG).

**Result.** Qwen correctly reported the outgoing ACH payment of 18,533.55 USD. It cited P03 and suggested checking whether the account had received funds in the previous 24 hours. The alert already recorded three incoming payments. Both source IDs were identifiable, though Qwen omitted the requested brackets.

GPT-4.1 mini accurately reported the amount and the prior payment counts, used bracketed IDs, and proposed checking whether funds were passed onward. That check added information a reviewer could investigate. P07 supports the beneficiary check. The short-interval detail comes from P03, which the note omitted. Neither passage mentions layering or circular movement.

**Decision.** On this alert, GPT-4.1 mini gave the more useful follow-up. A grounded version would ask the reviewer to compare payment times [P03] and inspect recipient details for evidence of onward transfers [P07]. One case is too little to choose a model for the workflow.

**Learning takeaways**

- Evaluate retrieval and writing separately. P03 gave a useful lead about timing. The note reduced it to a question the alert already answered.
- A JSON schema gave us two fields, `observation` and `follow_up`. Each still needed a human check for evidence and usefulness.
- For local inference, `num_ctx` sets the context window, `num_predict` caps generated tokens, and `keep_alive=0` unloads the model after the request. The first retry used its entire 128-token budget restating instructions. Asking for the two fields produced a complete note.
