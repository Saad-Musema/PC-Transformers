
from torch.utils.data import DataLoader, Subset
from Data_preprocessing.datasets.datasets import TokenizedDataset
from Data_preprocessing.config import Config
from utils.model_utils import pad_collate_fn, load_tokenizer

# Load tokenized datasets
train_dataset = TokenizedDataset("train", Config.TOKENIZER_DIR, Config.MAX_LENGTH)
train_dataset = Subset(train_dataset, range(min(len(train_dataset), 100000)))

valid_dataset = TokenizedDataset("valid", Config.TOKENIZER_DIR, Config.MAX_LENGTH)

test_dataset = TokenizedDataset("test", Config.TOKENIZER_DIR, Config.MAX_LENGTH)
test_dataset = Subset(test_dataset, range(min(len(test_dataset), 25000)))

# Load tokenizer
tokenizer = load_tokenizer()
pad_token_id = tokenizer.pad_token_id

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id)
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=Config.BATCH_SIZE,
    collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=Config.BATCH_SIZE,
    collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id)
) 
