import sys
from sys import platform

ON_COLAB = 'google.colab' in sys.modules
ON_CLUSTER = False if ON_COLAB else platform == "linux" or platform == "linux2"

MODEL_NAME = "bert-base-multilingual-cased"
DATASET = "squad"
MAX_LANG = 512
DOC_STRIDE = 256
OUT_FOLDER = "./temp"
MAX_TRAIN_SAMPLE = None if ON_COLAB or ON_CLUSTER else 8
MAX_EVAL_SAMPLE = None if ON_COLAB or ON_CLUSTER else 8
BATCH_SIZE_PRE_DEVICE_TRAIN = 12 if ON_COLAB else 20 if ON_CLUSTER else 2
BATCH_SIZE_PRE_DEVICE_EVAL = None
REPLACE_RATE = 0
OVER_WRITE = False
V_DATASET = "xquad"
V_DATASET_CONFIG = "xquad.ru"
LOSS_PRINT = 100
EVAL_PRINT = 4000
LR = 1e-5
WARM_UP = 2000
REPLACE_TABLE_PATH = None
EPOCH= 100
SEED = 1234

if not ON_CLUSTER and not ON_COLAB:
    run_str = f"python3 baseline_qa.py " \
              f"--model_name_or_path {MODEL_NAME} " \
              f"--dataset_name {DATASET} " \
              f"--max_seq_length {MAX_LANG} " \
              f"--doc_stride {DOC_STRIDE} " \
              f"--output_dir {OUT_FOLDER} " \
              f"--per_device_train_batch_size {BATCH_SIZE_PRE_DEVICE_TRAIN} " \
              f"--per_device_eval_batch_size " \
              f"{BATCH_SIZE_PRE_DEVICE_EVAL if BATCH_SIZE_PRE_DEVICE_EVAL is not None else BATCH_SIZE_PRE_DEVICE_TRAIN} " \
              f"--max_train_samples {MAX_TRAIN_SAMPLE} " \
              f"--max_eval_samples {MAX_EVAL_SAMPLE} "

elif ON_CLUSTER:
    # Fill String with paras here
    run_str = ""
    pass
elif ON_COLAB:
    run_str = f"python3 baseline_qa.py " \
              f"--model_name_or_path {MODEL_NAME} " \
              f"--dataset_name {DATASET} " \
              f"--max_seq_length {MAX_LANG} " \
              f"--doc_stride {DOC_STRIDE} " \
              f"--output_dir {OUT_FOLDER} " \
              f"--per_device_train_batch_size {BATCH_SIZE_PRE_DEVICE_TRAIN} " \
              f"--per_device_eval_batch_size " \
              f"{BATCH_SIZE_PRE_DEVICE_EVAL if BATCH_SIZE_PRE_DEVICE_EVAL is not None else BATCH_SIZE_PRE_DEVICE_TRAIN} " \
              f"--overwrite_cache {OVER_WRITE} " \
              f"--loss_interval {LOSS_PRINT} " \
              f"--eval_interval {EVAL_PRINT} " \
              f"--num_warmup_steps {WARM_UP} " \
              f"--learning_rate {LR} " \
              f"--num_train_epochs {EPOCH} " \
              f"--seed {SEED}"





else:
    raise ValueError(f"didn't except run {platform}, MAC,colab and linux only!")

print(run_str)
