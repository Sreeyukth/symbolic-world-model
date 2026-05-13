import torch

# Vocabulary — set by tokenizer at import time; overwritten in tokenizer.py
VOCAB_SIZE = None  # populated by Tokenizer.__init__

# Model dimensions
EMBED_DIM = 64
OP_EMBED_DIM = 32
GRU_HIDDEN = 128
GRU_LAYERS = 2
MAX_SEQ_LEN = 32
N_OPS = 4

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 60
LR_DECAY = True  # StepLR: halve LR every 20 epochs

# Device
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
