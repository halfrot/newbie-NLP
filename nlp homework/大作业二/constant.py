import torch

EPOCHS = 5
INPUT_SIZE = 50
HIDDEN_SIZE = 100
BATCH_SIZE = 1
LEARNING_RATE = 0.0001

STATE_FILE = "tmp.lstm"
WORD_TO_VECTOR_FILE = "ctb.50d.vec"
TRAIN_SET_FILE = "train.txt"
VALID_SET_FILE = "valid.txt"
VALID_PRECISION_FILE = "vp.txt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
