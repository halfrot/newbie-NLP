"""
ner_constant.py
用于设置全局常量
"""
STATE_FILE = "protected data/lstm.pth"
WORD_VECTOR_FILE = "ctb.50d.vec"
TRAIN_SET_FILE = "train_set.txt"
VALID_SET_FILE = "validation_set.txt"
VALID_F1_MEASURE_FILE = "valid_f1_measure.txt"
LOSS_FILE = "loss.txt"

SIZE = 50
BATCH_SIZE = 5
LEARNING_RATE = 0.001
EPOCHS = 5
"""
使用Attention based Bi-LSTM时，将BATCH_SIZE设为1
"""
