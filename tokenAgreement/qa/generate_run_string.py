import sys
from sys import platform

ON_COLAB = 'google.colab' in sys.modules
ON_CLUSTER = False if ON_COLAB else platform == "linux" or platform == "linux2"

MODEL_NAME = "xlm-roberta-base"
DATASET = "squad"
MAX_LANG = 384
DOC_STRIDE = 123
OUT_FOLDER = "./tmp/debug_squad"
MAX_TRAIN_SAMPLE = None if ON_COLAB or ON_CLUSTER else 8
MAX_EVAL_SAMPLE = None if ON_COLAB or ON_CLUSTER else 8
BATCH_SIZE_PRE_DEVICE_TRAIN = 16 if ON_COLAB else 20 if ON_CLUSTER else 2
BATCH_SIZE_PRE_DEVICE_EVAL = None
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

else:
    raise ValueError(f"didn't except run {platform}, MAC,colab and linux only!")


print(run_str)

