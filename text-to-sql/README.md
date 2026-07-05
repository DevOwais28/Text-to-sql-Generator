# Text-to-SQL with T5

Fine-tunes a T5-base transformer to translate natural language questions into SQL queries, using the Spider dataset (plus other Text-to-SQL datasets like ATIS/GeoQuery).

## Project Structure

```
text-to-sql/
├── README.md
├── requirements.txt
├── config.py                 # All hyperparameters & paths in one place
├── data/                     # Place train_spider.json, train_others.json, tables.json here
├── src/
│   ├── __init__.py
│   ├── schema_utils.py        # Builds readable schema strings from tables.json
│   ├── data_utils.py          # Loads data, train/val split, batching + tokenization
│   ├── train.py                # Training loop, validation, accuracy metric
│   └── predict.py              # Load trained model & generate SQL from a question
├── main.py                    # Entry point: trains the model end-to-end
└── checkpoints/                # Saved model weights go here (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
```

Download the Spider dataset (`train_spider.json`, `train_others.json`, `tables.json`) and place them inside `data/`.

## Training

```bash
python main.py
```

This will:
1. Load and merge the datasets
2. Build schema strings for each database
3. Split into train/val (90/10)
4. Fine-tune T5-base for the configured number of epochs
5. Save the best model (lowest validation loss) to `checkpoints/codet5_sql_simple.pth`

All hyperparameters (learning rate, batch size, epochs, paths) live in `config.py` — edit them there instead of the code.

## Predicting

Once you have a trained checkpoint:

```bash
python -m src.predict --db_id people_1 --question "how many people are there"
```

Or import it directly in Python:

```python
from src.predict import Predictor

predictor = Predictor(checkpoint_path="checkpoints/codet5_sql_simple.pth")
sql = predictor.predict("people_1", "list all names from people")
print(sql)
```

## Notes

- This was originally developed in Google Colab; `config.py` defaults to local paths but you can point `SAVE_PATH` at a mounted Google Drive path if running there.
- Training tracks token-level accuracy alongside loss. This is not the same as exact-match query accuracy — a model can get most tokens right while still producing an invalid or subtly wrong query.
