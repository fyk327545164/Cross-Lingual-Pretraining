import pickle
import random

from datasets import load_dataset
from transformers import AutoTokenizer
MODEL_NAME = "bert-base-multilingual-cased"
DATASET = "squad"
MAX_LANG = 512
DOC_STRIDE = 256
OUT_FOLDER = "./temp"
BATCH_SIZE_PRE_DEVICE_TRAIN = 6
BATCH_SIZE_PRE_DEVICE_EVAL = 6
REPLACE_RATE = 0
OVER_WRITE = False
LOSS_PRINT = 100
LR_FINE_TUNE = 1e-5
LR_PRE_TRAIN = 1e-5
WARM_UP = 2000
REPLACE_TABLE_PATH = None
EPOCH= 100
SEED = 1234
RATIO = 5
OUT_LOG_FILE = "./log"
EVLA_LANGE = "ar"
TABLE_PATH = f"table/aligned-tokens-en-{EVLA_LANGE}"
for EVLA_LANGE in ["ar","tr","zh","hi"]:
    for RATIO in [1,3,5,7,9]:

        raw_datasets = load_dataset(DATASET)
        train_dataset = raw_datasets["train"]
        question_column_name = "question"
        context_column_name = "context"
        answer_column_name = "answers"
        random_index = [0 for _ in range(RATIO)] + [1 for _ in range(10 - RATIO)]
        random.shuffle(random_index)
        with open(f"table/aligned-tokens-en-{EVLA_LANGE}", "rb") as fr:
            aligned_tokens_table = pickle.load(fr)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        pad_on_right = tokenizer.padding_side == "right"
        max_seq_length = MAX_LANG

        column_names = raw_datasets["train"].column_names
        def prepare_train_features(examples):
            # Some of the questions have lots of whitespace on the left, which is not useful and will make the
            # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
            # left whitespace
            examples["question"] = [q.lstrip() for q in examples[question_column_name]]
            original_questions = []
            original_contexts = []

            for idx in range(len(examples[context_column_name])):
                context = examples[context_column_name][idx]
                start = 0
                end = 0
                context_tokens = []
                context_tokens_idx = []
                is_space = context[start] == ' '
                while start < len(context):
                    new_is_space = context[start] == ' '
                    if new_is_space == is_space:
                        start += 1
                    else:
                        sub_string = context[end:start]
                        if len(sub_string.lstrip()) > 0:
                            context_tokens.append(sub_string)
                            context_tokens_idx.append(end)
                        is_space = new_is_space
                        end = start
                        start += 1
                context_tokens.append(context[end:start])
                context_tokens_idx.append(end)

                answers_text = examples[answer_column_name][idx]["text"][0]
                answers_text_token = answers_text.split()
                answers_text = " ".join(answers_text_token)
                answers_start = examples[answer_column_name][idx]["answer_start"][0]
                answer_range = (answers_start, answers_start + len(answers_text))

                question_tokens = examples[question_column_name][idx].split()
                original_questions.append(" ".join(question_tokens))
                for token_idx in range(len(question_tokens)):
                    cur_token = question_tokens[token_idx].lower().strip()
                    if cur_token in aligned_tokens_table:
                        if random.choice(random_index) == 0:
                            question_tokens[token_idx] = random.choice(aligned_tokens_table[cur_token])
                examples[question_column_name][idx] = " ".join(question_tokens)

                original_contexts.append(" ".join(context_tokens))
                for token_idx in range(len(context_tokens)):
                    cur_token, cur_token_idx = context_tokens[token_idx], context_tokens_idx[token_idx]
                    cur_token = cur_token.lower()
                    if not (answer_range[0] <= cur_token_idx <= answer_range[1]) and cur_token in aligned_tokens_table:
                        if random.choice(random_index) == 0:
                            context_tokens[token_idx] = random.choice(aligned_tokens_table[cur_token])

                examples[context_column_name][idx] = " ".join(context_tokens)
                new_answer_start = []

                if answers_text in examples[context_column_name][idx]:
                    new_answer_start.append(examples[context_column_name][idx].index(answers_text))
                else:
                    pass

                examples[answer_column_name][idx]["answer_start"] = new_answer_start

            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
            # in one example possible giving several features when a context is long, each of those features having a
            # context that overlaps a bit the context of the previous feature.
            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                return_offsets_mapping=True,
                padding="max_length",
            )
            tokenized_original = tokenizer(
                original_questions if pad_on_right else original_contexts,
                original_contexts if pad_on_right else original_questions,
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                padding="max_length",
            )
            tokenized_examples["eng_input_ids"] = tokenized_original['input_ids']
            tokenized_examples["eng_attention_mask"] = tokenized_original['attention_mask']
            tokenized_examples["eng_token_type_ids"] = tokenized_original['token_type_ids']

            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            # sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            # The offset mappings will give us a map from token to character position in the original context. This will
            # help us compute the start_positions and end_positions.
            offset_mapping = tokenized_examples.pop("offset_mapping")

            # Let's label those examples!
            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []

            for i, offsets in enumerate(offset_mapping):
                # We will label impossible answers with the index of the CLS token.
                input_ids = tokenized_examples["input_ids"][i]
                cls_index = input_ids.index(tokenizer.cls_token_id)
                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                # One example can give several spans, this is the index of the example containing this span of text.
                # sample_index = sample_mapping[i]
                answers = examples[answer_column_name][i]
                # If no answers are given, set the cls_index as answer.
                if len(answers["answer_start"]) == 0:
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Start/end character index of the answer in the text.
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])
                    # Start token index of the current span in the text.
                    token_start_index = 0
                    while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                        token_start_index += 1
                    # End token index of the current span in the text.
                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                        token_end_index -= 1
                    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                    if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                        tokenized_examples["start_positions"].append(cls_index)
                        tokenized_examples["end_positions"].append(cls_index)
                    else:
                        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                        # Note: we could go after the last offset if the answer is the last word (edge case).
                        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                            token_start_index += 1
                        tokenized_examples["start_positions"].append(token_start_index - 1)
                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        tokenized_examples["end_positions"].append(token_end_index + 1)
            return tokenized_examples


        train_dataset = train_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=1,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
        train_dataset.save_to_disk(f"/content/drive/MyDrive/CS546/data/squad_{EVLA_LANGE}_{RATIO}")