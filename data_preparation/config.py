import os
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent 
data_dir = Path(os.getenv("PC_DATA_DIR", base_dir / "data_preparation" / "data" / "tiny_shakespear"))
encoded_dir = Path(os.getenv("PC_ENCODED_DIR", base_dir / "data_preparation" / "encoded"))
tokenizer_path = base_dir / "data_preparation" / "tokenizer.json"

# Dataset files
train_path = data_dir / "train.csv"
valid_path = data_dir / "validation.csv"
test_path = data_dir / "test.csv"

# Tokenizer parameters
vocab_size = 1024
special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

# Training parameters
batch_size = int(os.getenv("PC_BATCH_SIZE", "8"))
max_len = int(os.getenv("PC_MAX_LEN", "208"))   # sequence length
