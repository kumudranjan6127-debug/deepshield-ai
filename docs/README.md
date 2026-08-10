# Documentation

Five documents, each answering one question. They are meant to be read in
this order by someone new, and consulted individually by someone who is not.

| Document | Answers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | How is it built, and what was left out on purpose? |
| [MODEL_CARD.md](MODEL_CARD.md) | What is the model, what was it trained on, what does it claim? |
| [BENCHMARK.md](BENCHMARK.md) | How accurate and how fast — and what has never been measured? |
| [LIMITATIONS.md](LIMITATIONS.md) | What can it not do? Read §1 before using it for anything. |
| [SECURITY.md](SECURITY.md) | What does it refuse, and what does it not defend against? |

Two more, for people working on it:

| Document | Answers |
|---|---|
| [CURRENT_STATE.md](CURRENT_STATE.md) | What is running right now, field by field |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | The ledger — every open issue, and every fixed one kept for the record |

---

## The rule these follow

**No number appears in this documentation that was not measured.** Where a
figure does not exist, the document says so and names the input that would
produce it — rather than leaving a gap that reads like an oversight, or
filling it with an estimate that reads like a fact.

`BENCHMARK.md`'s main table is generated:

```bash
python scripts/evaluate.py --data eval_data --report docs/BENCHMARK_TABLE.md
```

`DOCUMENTATION.md` predates most of the system and is superseded by the
files above; it is kept only for the frontend history it records.
