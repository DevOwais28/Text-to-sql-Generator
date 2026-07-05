# Text-to-SQL: Fine-Tuned T5 for Natural Language to SQL Translation

## Overview

This project fine-tunes a **T5-base** transformer model to translate natural language questions into SQL queries, using the **Spider dataset** — a large-scale, cross-domain benchmark spanning 100+ real-world relational database schemas (people, colleges, sports, insurance, etc.).

Given a question and a database schema, the model generates a syntactically correct, schema-grounded SQL query — without ever having seen that exact question during training.

## Approach

- **Base model:** `t5-base` (encoder-decoder transformer), fine-tuned end-to-end (not just a classification head)
- **Input format:** `translate English to SQL: <question> | schema: <table (col:type, col:type) | table (...)>`
- **Training data:** 8,659 examples from Spider + supplementary Text-to-SQL datasets, split 90/10 train/val
- **Training:** AdamW optimizer, gradient clipping, checkpointing on best validation loss, run for 7+ epochs

## Results

| Metric | Value (Epoch 7) |
|---|---|
| Train Loss | 0.0867 |
| Train Token Accuracy | 97.24% |
| Val Loss | 0.0881 |
| Val Token Accuracy | 97.58% |

Beyond token-level accuracy, the model was evaluated with **exact-match testing** on real validation examples across multiple schemas, correctly generating queries for:

- Simple `SELECT` statements
- `COUNT(*)` aggregates
- `MAX(...)` / `AVG(...)` aggregates
- Conditional `WHERE` filters

## Known Limitations

Digging past the headline accuracy number surfaced specific, explainable weaknesses rather than random failure:

1. **Zero-shot schemas:** On database schemas with zero training examples, the model produces plausible-looking but incorrect SQL (e.g., hallucinated syntax). This reflects the difficulty of true zero-shot schema generalization, not a training bug.
2. **Multi-column SELECT + GROUP BY:** The model occasionally drops the grouping column from the SELECT list when combined with aggregate functions.
3. **Table disambiguation:** On schemas with closely related tables (e.g., `course` vs `section`), the model sometimes queries the wrong-but-related table.
4. **Complex multi-table JOINs:** Like most base-size Text-to-SQL models, performance drops on Spider's "hard"/"extra-hard" queries requiring multiple JOINs and nested subqueries.

## Key Takeaway

Token-level accuracy (97%+) does not equal query correctness — a handful of wrong tokens in a short SQL query (a missing column, a hallucinated table name) can make an otherwise "97% accurate" prediction functionally useless. Exact-match / execution-based evaluation is necessary to understand real-world performance, especially for structured generation tasks like Text-to-SQL.

## Tech Stack

`Python` · `PyTorch` · `HuggingFace Transformers` · `T5` · `Google Colab`
