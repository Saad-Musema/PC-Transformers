import os
import torch
import time
import logging
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import train_path, valid_path, test_path, tokenizer_path, encoded_dir, vocab_size, special_tokens

"""
Usage: python prepare_tokens.py
"""
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

SMOKE_TEST = os.getenv("PC_SMOKE_TEST") == "1"
DEFAULT_SMOKE_PERCENT = os.getenv("PC_SMOKE_DATA_PERCENT", "1.0")
SMOKE_SPLIT_PERCENTAGES = {
    "train": os.getenv("PC_SMOKE_TRAIN_PERCENT", DEFAULT_SMOKE_PERCENT),
    "valid": os.getenv("PC_SMOKE_VALID_PERCENT", DEFAULT_SMOKE_PERCENT),
    "test": os.getenv("PC_SMOKE_TEST_PERCENT", DEFAULT_SMOKE_PERCENT),
}


def load_split_text(split_name, path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    text = path.read_text(encoding="utf-8")
    if SMOKE_TEST:
        percent = float(SMOKE_SPLIT_PERCENTAGES[split_name])
        if percent <= 0:
            raise ValueError(f"Smoke dataset percentage for {split_name} must be positive, got {percent}")
        char_count = max(1, int(len(text) * (percent / 100.0)))
        logger.info(
            "Smoke mode truncating %s split to %.3f%% of source text (%s chars)",
            split_name,
            percent,
            char_count,
        )
        return text[:char_count]
    return text


@contextmanager
def tokenizer_training_path():
    if not SMOKE_TEST:
        yield train_path
        return

    train_text = load_split_text("train", train_path)
    with TemporaryDirectory(prefix="pc-transformers-smoke-") as temp_dir:
        temp_train_path = Path(temp_dir) / train_path.name
        temp_train_path.write_text(train_text, encoding="utf-8")
        yield temp_train_path

def build_tokenizer():
    """ This function trains a BPE tokenizer on the given dataset and saves it."""
    tokenizer = Tokenizer(BPE(unk_token = special_tokens[0]))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        special_tokens = special_tokens,
        vocab_size = vocab_size
        )

    _ = load_split_text("valid", valid_path)
    _ = load_split_text("test", test_path)

    start_time = time.perf_counter()
    with tokenizer_training_path() as training_path:
        tokenizer.train(files=[str(training_path)], trainer=trainer)
    elapsed = time.perf_counter() - start_time

    tokenizer.save(str(tokenizer_path))
    logger.info(f"Tokenizer saved at {tokenizer_path}")
    logger.info(f"Tokenizer training took {elapsed:.2f} seconds.")

    return tokenizer

def encode_and_save(tokenizer):
    """
    This function takes the raw text from train/valid/test, uses the tokenizer to convert words 
    (or subwords) into token IDs, and saves them as PyTorch tensors (.pt files).
    """
    encoded_dir.mkdir(exist_ok=True, parents=True)

    splits = {
        "train": train_path,
        "valid": valid_path,
        "test": test_path,
    }

    for split_name, path in splits.items():
        text = load_split_text(split_name, path)

        start_time = time.perf_counter()
        encoded_ids = tokenizer.encode(text).ids  
        elapsed = time.perf_counter() - start_time

        tensor = torch.tensor(encoded_ids, dtype=torch.long)
        save_path = encoded_dir / f"{split_name}.pt"
        torch.save(tensor, save_path)

        logger.info(f"Saved encoded {split_name} dataset in {save_path}")
        logger.info(f"Tokenizing {split_name} split took {elapsed:.2f} seconds.")

if __name__ == "__main__":
    tokenizer = build_tokenizer()
    encode_and_save(tokenizer)
