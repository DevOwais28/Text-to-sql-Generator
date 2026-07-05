from transformers import T5ForConditionalGeneration, AutoTokenizer

import config
from src.data_utils import load_raw_data, build_input_target_pairs, train_val_split
from src.train import train_model


def main():
    print(f"Using device: {config.DEVICE}")

    # ── 1. Load data & build schemas ────────────────────────────────
    print("Loading data...")
    data_df, db_schemas = load_raw_data()

    # ── 2. Build (input, target) pairs ──────────────────────────────
    all_inputs, all_targets = build_input_target_pairs(data_df, db_schemas)

    # ── 3. Train/val split ──────────────────────────────────────────
    train_src, train_tgt, val_src, val_tgt = train_val_split(all_inputs, all_targets)
    print(f"Train: {len(train_src)} | Val: {len(val_src)}")

    # ── 4. Load model & tokenizer ────────────────────────────────────
    print(f"Loading {config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME).to(config.DEVICE)
    print(f"Model ready on {config.DEVICE}")

    # ── 5. Train ──────────────────────────────────────────────────────
    train_model(model, tokenizer, train_src, train_tgt, val_src, val_tgt, config.DEVICE)

    print(f"\nTraining complete. Best model saved to {config.SAVE_PATH}")


if __name__ == "__main__":
    main()
