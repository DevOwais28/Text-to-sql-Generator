import random
import pandas as pd

from src.schema_utils import build_schemas
import config


def load_raw_data():
    """Loads and merges the Spider + other datasets, and builds schema strings."""
    df_spider = pd.read_json(config.SPIDER_PATH)
    df_other  = pd.read_json(config.OTHERS_PATH)
    data_df   = pd.concat([df_spider, df_other], ignore_index=True)

    db_schemas = build_schemas(config.TABLES_PATH)

    return data_df, db_schemas


def build_input_target_pairs(data_df: pd.DataFrame, db_schemas: dict):
    """
    Converts each row into:
        input : "translate English to SQL: <question> | schema: <schema>"
        target: "<SQL query>"
    """
    all_inputs, all_targets = [], []

    for _, row in data_df.iterrows():
        schema = db_schemas.get(row["db_id"], "")
        src = f"translate English to SQL: {row['question']} | schema: {schema}"
        tgt = row["query"]

        all_inputs.append(src)
        all_targets.append(tgt)

    return all_inputs, all_targets


def train_val_split(all_inputs, all_targets, split_ratio=None, seed=None):
    """Shuffles and splits (inputs, targets) into train/val sets."""
    split_ratio = split_ratio or config.TRAIN_VAL_SPLIT
    seed        = seed if seed is not None else config.RANDOM_SEED

    random.seed(seed)
    combined = list(zip(all_inputs, all_targets))
    random.shuffle(combined)
    all_inputs, all_targets = zip(*combined)

    split = int(split_ratio * len(all_inputs))

    train_src, val_src = all_inputs[:split], all_inputs[split:]
    train_tgt, val_tgt = all_targets[:split], all_targets[split:]

    return train_src, train_tgt, val_src, val_tgt


def get_batch(src_list, tgt_list, i, batch_size, tokenizer, device):
    """
    Tokenizes one batch of (input, target) string pairs.

    Returns:
        input_ids, attention_mask, labels (padding replaced with -100)
    """
    src_batch = src_list[i: i + batch_size]
    tgt_batch = tgt_list[i: i + batch_size]

    src_enc = tokenizer(
        list(src_batch),
        max_length=config.MAX_SRC_LEN,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    tgt_enc = tokenizer(
        list(tgt_batch),
        max_length=config.MAX_TGT_LEN,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    labels = tgt_enc["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return src_enc["input_ids"], src_enc["attention_mask"], labels
