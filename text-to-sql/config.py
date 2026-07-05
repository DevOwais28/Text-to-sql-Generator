import os
import torch

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")

SPIDER_PATH = os.path.join(DATA_DIR, "train_spider.json")
OTHERS_PATH = os.path.join(DATA_DIR, "train_others.json")
TABLES_PATH = os.path.join(DATA_DIR, "tables.json")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
SAVE_PATH      = os.path.join(CHECKPOINT_DIR, "codet5_sql_simple.pth")

# If running in Google Colab with Drive mounted, override like this in main.py:
# config.SAVE_PATH = "/content/drive/MyDrive/codet5_sql_simple.pth"

# ── Model ─────────────────────────────────────────────────────────────
MODEL_NAME = "t5-base"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Training hyperparameters ─────────────────────────────────────────
LEARNING_RATE = 5e-5
BATCH_SIZE    = 8
EPOCHS        = 10
MAX_GRAD_NORM = 1.0

# ── Tokenization ──────────────────────────────────────────────────────
MAX_SRC_LEN = 512   # question + schema
MAX_TGT_LEN = 128   # SQL query

# ── Generation (for predict.py) ──────────────────────────────────────
MAX_NEW_TOKENS       = 64
NUM_BEAMS            = 4
NO_REPEAT_NGRAM_SIZE = 2

# ── Misc ──────────────────────────────────────────────────────────────
TRAIN_VAL_SPLIT = 0.9   # 90% train, 10% val
RANDOM_SEED     = 42
LOG_EVERY_N_STEPS = 200
