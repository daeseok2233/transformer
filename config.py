from pathlib import Path

# ---- Paths ----
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_JSON = DATA_DIR / "일상생활구어체_train_set.json"
VALID_JSON = DATA_DIR / "일상생활구어체_valid_set.json"

# ---- Data Keys ----
SRC_KEY = "ko"
TGT_KEY = "mt"

# ---- Hyperparams ----
SEED = 42
BATCH_SIZE = 32
MAX_LENGTH = 64
TRAIN_MAX_SAMPLES = 200   # None 이면 전체
VALID_MAX_SAMPLES = 30    # None 이면 전체

# ---- DataLoader ----
NUM_WORKERS = 2
PIN_MEMORY = True

# ---- Device policy (optional): "auto" | "cpu" | "cuda"
DEVICE_POLICY = "auto"