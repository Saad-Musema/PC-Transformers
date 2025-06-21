import os

class Config:
    VOCAB_SIZE = None
    PAD_ID=None
    EOS_ID=None
    MAX_LENGTH = 128
        
    DATASET_NAME = "ptb"  # "ptb" for penntreebank "opwb" for OpenWebText
    
    # Data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    DATA_DIR = os.path.join(BASE_DIR, "Data") 
    TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer", "outputs")  
   
    # Training
    BATCH_SIZE = 32

