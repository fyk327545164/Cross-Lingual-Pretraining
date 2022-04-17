from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AdamW, \
    default_data_collator
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

np.random.seed(1234)
torch.manual_seed(1234)
random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', required=True, type=int)
parser.add_argument('--lg', required=True, type=str)
parser.add_argument('--mode', required=True, type=str)

args = parser.parse_args()

metric = load_metric("seqeval")

global_label_list = ["S" + str(n) for n in range(17)]

LR_PRETRAIN = 0.00001
LR_FINETUNE = 0.00009
WARMUP_STEPS = 100

MODEL_NAME = "bert-base-multilingual-cased"

aligned_tokens = {}
for _lg in ["hi", "de", "ru", "tr"]:
    with open("data/aligned-tokens-en-{}".format(_lg), "rb") as fr:
        _aligned_tokens = pickle.load(fr)
        for key, val in tqdm(_aligned_tokens.items()):
            if key not in aligned_tokens:
                aligned_tokens[key] = []
            aligned_tokens[key] += val[:]

random_index = [0 for _ in range(args.ratio)] + [1 for _ in range(100 - args.ratio)]
random.shuffle(random_index)
print(random_index)


class CrossLingualModel(nn.Module):
    def __init__(self, num_labels=len(global_label_list)):
        super().__init__()

        self.xlm = BertModel.from_pretrained(MODEL_NAME)

        self.hidden_size = 768
        self.dropout = nn.Dropout(0.1)

        self.num_labels = num_labels
        self.ner_classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.loss = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, ner_labels=None, eng_input_ids=None,
                eng_attention_mask=None, eng_labels=None, **kwargs):
        input_ids = input_ids.to(self.xlm.device)
        attention_mask = attention_mask.to(self.xlm.device)
        if eng_input_ids is not None:
            eng_input_ids = eng_input_ids.to(self.xlm.device) if args.mode == "align" else None
            eng_attention_mask = eng_attention_mask.to(self.xlm.device) if args.mode == "align" else None

        outputs = self.xlm(input_ids, attention_mask, eng_input_ids=eng_input_ids,
                           eng_attention_mask=eng_attention_mask)

        sequence_output = self.dropout(outputs[0])
        logits = self.ner_classifier(sequence_output)

        loss = None
        if ner_labels is not None:
            ner_labels = ner_labels.to(self.xlm.device)
            if eng_labels is not None and args.mode == "align":
                eng_labels = eng_labels.to(self.xlm.device)
                ner_labels = torch.cat((ner_labels, eng_labels), 1)

            loss = self.loss(logits.view(-1, self.num_labels), ner_labels.view(
                -1))  # + self.loss(eng_logits.view(-1, self.num_labels), eng_labels.view(-1))

        return loss, logits


LANGUAGE_IDS = ["Hindi", "German", "Russian", "Turkish"]


def _process_task_data(lg):
    raw_datasets = load_dataset("xtreme", "udpos." + lg,
                                cache_dir="/brtx/605-nvme1/yukunfeng/cross-lingual/tokenAgreement/data/huggingface")  # ["train_en", "val_en"])

    column_names = raw_datasets["test"].column_names
    features = raw_datasets["test"].features

    text_column_name = "tokens"
    label_column_name = "pos_tags"
    lang_column_name = "lang"

    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    label_list = get_label_list(raw_datasets["test"][label_column_name])
    label_to_id = {l: i for i, l in enumerate(label_list)}

    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, add_prefix_space=True)

    # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True, add_prefix_space=True)
    # Map that sends B-Xxx label to its I-Xxx counterpart

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=128,
            padding="max_length",
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    # print(label)
                    # print(word_idx)
                    label_ids.append(label[word_idx])

            labels.append(label_ids)
        tokenized_inputs["ner_labels"] = labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["test"].column_names,
        desc="Running tokenizer on dataset",
    )
    return processed_raw_datasets["test"]


def process_task_data():
    eval_dataloaders = {}
    for lg in LANGUAGE_IDS:
        eval_dataloaders[lg] = DataLoader(_process_task_data(lg), collate_fn=default_data_collator,
                                          batch_size=64)

    return eval_dataloaders


# https://github.com/Babelscape/wikineural
def get_train_dataloader():
    raw_datasets = load_dataset("xtreme", "udpos.English", keep_in_memory=True,
                                cache_dir="/brtx/605-nvme1/yukunfeng/cross-lingual/tokenAgreement/data/huggingface")  # ["train_en", "val_en"])

    train_en_tokens = raw_datasets["train"]["tokens"][:]

    for i in trange(len(train_en_tokens)):
        for token_i in range(len(train_en_tokens[i])):
            cur_token = train_en_tokens[i][token_i].lower()
            if cur_token in aligned_tokens:
                if random.choice(random_index) == 0:
                    train_en_tokens[i][token_i] = random.choice(aligned_tokens[cur_token])
    raw_datasets["train"] = raw_datasets["train"].add_column("new_tokens", train_en_tokens)

    column_names = raw_datasets["train"].column_names
    features = raw_datasets["train"].features

    for key in list(raw_datasets.keys()):
        if key != "train":
            del raw_datasets[key]

    text_column_name = "new_tokens"
    label_column_name = "pos_tags"
    lang_column_name = "lang"

    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    label_list = get_label_list(raw_datasets["train"][label_column_name])
    label_to_id = {l: i for i, l in enumerate(label_list)}

    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, add_prefix_space=True)

    # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True, add_prefix_space=True)
    # Map that sends B-Xxx label to its I-Xxx counterpart

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["new_tokens"],
            max_length=128,
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
        )

        original_tokenized_inputs = tokenizer(
            examples["tokens"],
            max_length=128,
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(label_to_id[label[word_idx]])

            labels.append(label_ids)
        tokenized_inputs["ner_labels"] = labels

        if args.ratio == 0:
            return tokenized_inputs

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = original_tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                else:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                # else:
                #    label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])

                # previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["eng_input_ids"] = original_tokenized_inputs["input_ids"]
        tokenized_inputs["eng_attention_mask"] = original_tokenized_inputs["attention_mask"]
        tokenized_inputs["eng_labels"] = labels

        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        keep_in_memory=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_raw_datasets["train"]
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=64
    )

    return train_dataloader


class ReverseSqrtScheduler:
    def __init__(self, optimizer, lr, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

        self.decay_factor = [_lr * n_warmup_steps ** 0.5 for _lr in lr]
        self.lr_step = [(_lr - 0) / n_warmup_steps for _lr in lr]

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _update_learning_rate(self):
        self.n_steps += 1
        if self.n_steps < self.n_warmup_steps:
            lr = [self.n_steps * _lr for _lr in self.lr_step]
        else:
            lr = [_decay_factor * self.n_steps ** -0.5 for _decay_factor in self.decay_factor]

        for i, param_group in enumerate(self._optimizer.param_groups):
            param_group['lr'] = lr[i]


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

    with open("results", 'a') as fw:
        fw.write('----------------------\n')

    # ith torch.no_grad():
    #    model.eval()
    #    evaluate(model, eval_dataloaders)

    for epoch in range(7):
        model.train()
        all_loss = 0
        update_step = 0
        train_dataloader = get_train_dataloader()
        print("train")
        for batch in tqdm(train_dataloader):
            loss, _ = model(**batch)
            all_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            update_step += 1

        print("epoch: {}, Update Steps {}, loss: {}\n".format(epoch, update_step, all_loss / update_step))
        with torch.no_grad():
            model.eval()
            evaluate(model, eval_dataloaders)


def compute_metrics(p):
    predictions, labels = p
    label_list = global_label_list

    true_predictions = [
        [label_list[int(p)] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[int(l)] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    res = {"precision": results["overall_precision"],
           "recall": results["overall_recall"],
           "f1": results["overall_f1"],
           "accuracy": results["overall_accuracy"]}
    with open("results", 'a') as fw:
        fw.write('{} \n'.format(res))
    print(res)


def evaluate(model, dataloaders):
    for lg, dataloader in dataloaders.items():
        with open("results", 'a') as fw:
            fw.write('{}-{} \n'.format(lg, args.ratio))
        print(lg)
        _evaluate(model, dataloader, lg)


def _evaluate(model, dataloader, lg):
    preds = []
    targets = []

    for batch in tqdm(dataloader):
        label_ids = batch["ner_labels"].view(-1, 1).tolist()

        _, logits = model(batch["input_ids"], batch["attention_mask"])
        logits = logits.cpu()
        index = torch.argmax(logits, -1).view(-1, 1).tolist()

        preds += index[:]
        targets += label_ids[:]

    compute_metrics((preds, targets))


main()