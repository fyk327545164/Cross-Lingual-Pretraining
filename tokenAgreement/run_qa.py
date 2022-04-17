from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AdamW, \
    default_data_collator, EvalPrediction
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from datasets import load_metric, ClassLabel
import torch.utils.data.distributed
from datasets import Dataset

from datasets import load_dataset
import pickle
import random

import argparse
import numpy as np
from collections import Counter
from bert import BertModel
from utils_qa import postprocess_qa_predictions

np.random.seed(1234)
torch.manual_seed(1234)
random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', required=True, type=int)
parser.add_argument('--lg', required=True, type=str)
parser.add_argument('--mode', required=True, type=str)

args = parser.parse_args()

metric = load_metric("squad")

LR_PRETRAIN = 0.00005
LR_FINETUNE = 0.0001
WARMUP_STEPS = 200

MODEL_NAME = "bert-base-multilingual-cased"

aligned_tokens = {}
for _lg in ["hi", "de","ru","tr"]:
    with open("data/aligned-tokens-en-{}".format(_lg), "rb") as fr:
        _aligned_tokens = pickle.load(fr)
        for key, val in tqdm(_aligned_tokens.items()):
            if key not in aligned_tokens:
                aligned_tokens[key] = []
            aligned_tokens[key] += val[:]
            # ndom.shuffle(aligned_tokens[key])

random_index = [0 for _ in range(args.ratio)] + [1 for _ in range(100 - args.ratio)]
random.shuffle(random_index)
print(args.ratio)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, add_prefix_space=True)


class CrossLingualModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.xlm = BertModel.from_pretrained(MODEL_NAME)

        self.hidden_size = 768
        self.dropout = nn.Dropout(0.1)

        self.qa_classifier = nn.Linear(self.hidden_size, 2)

        self.loss = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None, eng_input_ids=None,
                eng_attention_mask=None, eng_start_positions=None, eng_end_positions=None, **kwargs):
        input_ids = input_ids.to(self.xlm.device)
        attention_mask = attention_mask.to(self.xlm.device)
        if eng_input_ids is not None:
            eng_input_ids = eng_input_ids.to(self.xlm.device) if args.mode == "align" else None
            eng_attention_mask = eng_attention_mask.to(self.xlm.device) if args.mode == "align" else None

        outputs = self.xlm(input_ids, attention_mask, eng_input_ids=eng_input_ids,
                           eng_attention_mask=eng_attention_mask)

        sequence_output = self.dropout(outputs[0])
        logits_all = self.qa_classifier(sequence_output)

        logits = logits_all[:, :384]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None:
            start_positions = start_positions.to(self.xlm.device)
            end_positions = end_positions.to(self.xlm.device)
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

            if eng_input_ids is not None:
                eng_start_positions = eng_start_positions.to(self.xlm.device)
                eng_end_positions = eng_end_positions.to(self.xlm.device)
                if len(eng_start_positions.size()) > 1:
                    eng_start_positions = eng_start_positions.squeeze(-1)
                if len(eng_end_positions.size()) > 1:
                    eng_end_positions = eng_end_positions.squeeze(-1)

                eng_logits = logits_all[:, 384:]
                eng_start_logits, eng_end_logits = eng_logits.split(1, dim=-1)
                eng_start_logits = eng_start_logits.squeeze(-1).contiguous()
                eng_end_logits = eng_end_logits.squeeze(-1).contiguous()

                ignored_index = eng_start_logits.size(1)
                eng_start_positions = eng_start_positions.clamp(0, ignored_index)
                eng_end_positions = eng_end_positions.clamp(0, ignored_index)
                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(eng_start_logits, eng_start_positions)
                end_loss = loss_fct(eng_end_logits, eng_end_positions)
                eng_loss = (start_loss + end_loss) / 2

                loss = (loss + eng_loss) / 2

        return loss, start_logits, end_logits


LANGUAGE_IDS = ["hi", "de", "ru", "tr"]


def _process_task_data(lg):
    max_seq_length = 384

    valid_datasets = load_dataset("xquad", f"xquad.{lg}",
                                cache_dir="/brtx/605-nvme1/yukunfeng/cross-lingual/tokenAgreement/data/huggingface")  # ["train_en", "val_en"])

    # column_names = valid_datasets["train"].column_names
    valid_column_names = valid_datasets["validation"].column_names
    question_column_name = "question"# if "question" in column_names else column_names[0]
    context_column_name = "context"# if "context" in column_names else column_names[1]
    answer_column_name = "answers"# if "answers" in column_names else column_names[2]


    pad_on_right = tokenizer.padding_side == "right"

    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    eval_dataset = valid_datasets.map(
            prepare_validation_features,
            batched=True,
            remove_columns=valid_column_names,
            desc="Running tokenizer on validation dataset",
        )

    return valid_datasets["validation"], eval_dataset["validation"]


def process_task_data():
    eval_dataloaders = {}
    for lg in LANGUAGE_IDS:
        eval_examples, eval_dataset = _process_task_data(lg)
        eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
        eval_dataloaders[lg] = (DataLoader(eval_dataset_for_model, collate_fn=default_data_collator,
                                          batch_size=32), eval_examples, eval_dataset)

    return eval_dataloaders


# https://github.com/Babelscape/wikineural
def get_train_dataloader():
    raw_datasets = load_dataset("squad",
                                cache_dir="/brtx/605-nvme1/yukunfeng/cross-lingual/tokenAgreement/data/huggingface")  # ["train_en", "val_en"])

    train_dataset = raw_datasets["train"]

    column_names = raw_datasets["train"].column_names
    valid_column_names = raw_datasets["validation"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    pad_on_right = tokenizer.padding_side == "right"

    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # print(len(examples[question_column_name]))
        # print(len(examples[context_column_name]))

        eng_tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=384,
            # stride=128,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = eng_tokenized_examples.pop("offset_mapping")
        # Let's label those examples!
        eng_tokenized_examples["start_positions"] = []
        eng_tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = eng_tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = eng_tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = i
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                eng_tokenized_examples["start_positions"].append(cls_index)
                eng_tokenized_examples["end_positions"].append(cls_index)
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
                    eng_tokenized_examples["start_positions"].append(cls_index)
                    eng_tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    eng_tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    eng_tokenized_examples["end_positions"].append(token_end_index + 1)

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
            for token_idx in range(len(question_tokens)):
                cur_token = question_tokens[token_idx].lower()
                if cur_token in aligned_tokens:
                    if random.choice(random_index) == 0:
                        question_tokens[token_idx] = random.choice(aligned_tokens[cur_token])
            examples[question_column_name][idx] = " ".join(question_tokens)

            orginal_context = examples[context_column_name][idx][:]

            for token_idx in range(len(context_tokens)):
                cur_token, cur_token_idx = context_tokens[token_idx], context_tokens_idx[token_idx]
                cur_token = cur_token.lower()
                if not (answer_range[0] <= cur_token_idx <= answer_range[1]) and cur_token in aligned_tokens:
                    if random.choice(random_index) == 0:
                        context_tokens[token_idx] = random.choice(aligned_tokens[cur_token])
            examples[context_column_name][idx] = " ".join(context_tokens)

            new_answer_start = []

            if answers_text in examples[context_column_name][idx]:
                new_answer_start.append(examples[context_column_name][idx].index(answers_text))
            else:
                print(f"idx: {idx}, true answer: {answers_text}\n")
                print(f"idx: {idx}, context: {examples[context_column_name][idx]}\n")
                print(f"idx: {idx}, answer: {examples[answer_column_name][idx]}\n")
                print(f"idx: {idx}, orignal: {orginal_context}\n")

            examples[answer_column_name][idx]["answer_start"] = new_answer_start

        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=384,
            # stride=128,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        # sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        previous_index = -1

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = i

            answers = examples[answer_column_name][sample_index]
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
        tokenized_examples["eng_input_ids"] = eng_tokenized_examples["input_ids"]
        tokenized_examples["eng_attention_mask"] = eng_tokenized_examples["attention_mask"]
        tokenized_examples["eng_start_positions"] = eng_tokenized_examples["start_positions"]
        tokenized_examples["eng_end_positions"] = eng_tokenized_examples["end_positions"]

        return tokenized_examples

    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        keep_in_memory=True,
        remove_columns=column_names,
        desc="Running tokenizer on train dataset",
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=20
    )

    return train_dataloader


def main():
    eval_dataloaders = process_task_data()

    model = CrossLingualModel()
    model.cuda()
    pretrained_params = []
    finetune_params = []
    for (name, p) in model.named_parameters():
        if "xlm" in name:
            pretrained_params.append(p)
        else:
            finetune_params.append(p)

    optimizer = AdamW(
        [{'params': pretrained_params, 'lr': LR_PRETRAIN}, {'params': finetune_params, 'lr': LR_FINETUNE}])
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS,
                                                6000)  # ReverseSqrtScheduler(optimizer, [LR_PRETRAIN, LR_FINETUNE], WARMUP_STEPS)
    # optimizer = AdamW(model.parameters(), LR_PRETRAIN)
    # scheduler = ReverseSqrtScheduler(optimizer, [LR_PRETRAIN], WARMUP_STEPS)


    # evaluate(model, eval_dataloaders)

    for epoch in range(5):
        model.train()
        all_loss = 0
        update_step = 0
        train_dataloader = get_train_dataloader()
        for batch in tqdm(train_dataloader):
            loss, _, _ = model(**batch)
            all_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            update_step += 1

        print("epoch: {}, Update Steps {}, loss: {}\n".format(epoch, update_step, all_loss / update_step))
        evaluate(model, eval_dataloaders)


def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=False,
            n_best_size=20,
            max_answer_length=30,
            null_score_diff_threshold=0.0,
            output_dir="./",
            prefix=stage,
        )
        # Format the result to the format the metric expects.

        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step: step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat


def evaluate(model, dataloaders):
    for lg, (dataloader, eval_examples, eval_dataset) in dataloaders.items():
        print(lg)
        _evaluate(model, dataloader, eval_examples, eval_dataset)


def _evaluate(model, dataloader, eval_examples, eval_dataset):

    model.eval()
    all_start_logits = []
    all_end_logits = []
    for step, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            _, start_logits, end_logits  = model(**batch)

            all_start_logits.append(start_logits.cpu().numpy())
            all_end_logits.append(end_logits.cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

    start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)
    eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
    print(f"Evaluation metrics: {eval_metric}")

    model.train()


main()