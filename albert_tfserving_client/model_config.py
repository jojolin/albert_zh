from os import path

ALBERT_DIR = path.dirname(path.abspath(__file__))

VOCAB_FILE = path.join(ALBERT_DIR, 'vocab.txt')

LABEL_CLASS_FILE = path.join(ALBERT_DIR, 'label_class.txt')

MAX_SEQ_LENGTH = 128

MODEL_NAME = "albert_base_remy_lac_withcls_cooking_others"

HOST_PORT_HTTP = "192.168.33.25:8511"

HOST_PORT_GRPC = "192.168.33.25:8510"
