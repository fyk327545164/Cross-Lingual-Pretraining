from transformers import AutoTokenizer, AdamW, default_data_collator, XLMRobertaModel, BertModel, BatchEncoding
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

np.random.seed(1234)
torch.manual_seed(1234)
random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', required=True, type=int)
args = parser.parse_args()

metric = load_metric("seqeval")

global_label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

LR_PRETRAIN = 0.00008
LR_FINETUNE = 0.00008
WARMUP_STEPS = 500

MODEL_NAME = "bert-base-multilingual-cased"
import os

with open("aligned_tokens", "rb") as fr:
    aligned_tokens = pickle.load(fr)

# aligned_index = {}
# for key, val in _aligned_index.items():
#     aligned_index[key] = [Counter(val).most_common()[0][0]]

random_index = [0 for _ in range(args.ratio)] + [1 for _ in range(10 - args.ratio)]
random.shuffle(random_index)
print(random_index)


# random.seed(1234)

def data_collator(features):
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    for f in features:
        _input_ids = f["input_ids"]
        input_ids = []
        for _id in _input_ids:
            if _id in aligned_index:
                if random.choice(random_index) == 0:
                    new_index = random.choice(aligned_index[_id])
                    input_ids.append(new_index)
                    continue
            input_ids.append(_id)
        f["input_ids"] = input_ids

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


class CrossLingualModel(nn.Module):
    def __init__(self, num_labels=len(global_label_list)):
        super().__init__()

        self.xlm = BertModel.from_pretrained(MODEL_NAME)

        self.hidden_size = 768
        self.dropout = nn.Dropout(0.1)

        self.num_labels = num_labels
        self.ner_classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.loss = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, ner_labels=None, **kwargs):
        input_ids = input_ids.to(self.xlm.device)
        attention_mask = attention_mask.to(self.xlm.device)
        # rint(input_ids.size())
        outputs = self.xlm(input_ids, attention_mask)

        sequence_output = self.dropout(outputs[0])
        logits = self.ner_classifier(sequence_output)

        loss = None
        if ner_labels is not None:
            ner_labels = ner_labels.to(self.xlm.device)
            loss = self.loss(logits.view(-1, self.num_labels), ner_labels.view(-1))

        return loss, logits


LANGUAGE_IDS = ["en", "ru"]  # , "es", "nl", "pl", "ru", "it", "fr", "pt"]

lg_set = {"val_en", "val_ru", "train_en"}


# https://github.com/Babelscape/wikineural
def process_task_data():
    raw_datasets = load_dataset("Babelscape/wikineural",
                                cache_dir="/brtx/605-nvme1/yukunfeng/cross-lingual/tokenAgreement/data/huggingface")  # ["train_en", "val_en"])

    column_names = raw_datasets["train_en"].column_names
    features = raw_datasets["train_en"].features

    for key in list(raw_datasets.keys()):
        if key not in lg_set:
            del raw_datasets[key]

    text_column_name = "tokens"
    label_column_name = "ner_tags"
    lang_column_name = "lang"

    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.
    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train_en"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}

    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, add_prefix_space=True)
    # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True, add_prefix_space=True)
    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label >= 1 and label % 2 == 1:
            b_to_i_label.append(idx + 1)
        else:
            b_to_i_label.append(idx)

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
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])

                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["ner_labels"] = labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train_en"].column_names,
        desc="Running tokenizer on dataset",
    )

    # train_dataset = processed_raw_datasets["train_en"]
    # train_dataloader = DataLoader(
    #     train_dataset, shuffle=True, collate_fn=data_collator, batch_size=32
    # )

    eval_dataloaders = {}
    for lg in LANGUAGE_IDS:
        eval_dataloaders[lg] = DataLoader(processed_raw_datasets["val_{}".format(lg)], collate_fn=default_data_collator,
                                          batch_size=64)

    return eval_dataloaders


# https://github.com/Babelscape/wikineural
def get_train_dataloader():
    raw_datasets = load_dataset("Babelscape/wikineural", keep_in_memory=True,
                                cache_dir="/brtx/605-nvme1/yukunfeng/cross-lingual/tokenAgreement/data/huggingface")  # ["train_en", "val_en"])
    train_en_tokens = raw_datasets["train_en"]["tokens"][:]
    train_en_labels = raw_datasets["train_en"]["ner_tags"][:]

    for i in trange(len(train_en_tokens)):
        for token_i in range(len(train_en_tokens[i])):
            cur_token = train_en_tokens[i][token_i].lower()
            if cur_token in aligned_tokens:
                if random.choice(random_index) == 0:
                    # rint("---------")
                    train_en_tokens[i][token_i] = random.choice(aligned_tokens[cur_token])
    raw_datasets["train_en"] = raw_datasets["train_en"].add_column("new_tokens", train_en_tokens)
    # rint(raw_datasets.column_names)
    # print(raw_datasets["train_en"].column_names)
    # raw_datasets["train_en"].add_column(new_tokens",train_en_tokens)
    # print(raw_datasets["train_en"]["new_tokens"][10], train_en_tokens[10])
    # print(raw_datasets["train_en"]["new_tokens"][100], train_en_tokens[100])
    # print(raw_datasets["train_en"]["new_tokens"][200], train_en_tokens[200])
    # print(raw_datasets["train_en"]["new_tokens"][300],train_en_tokens[300])
    # print(raw_datasets["train_en"]["new_tokens"][400], train_en_tokens[400])

    # raw_datasets = Dataset.from_dict({"train_en":{"tokens": train_en_tokens, "ner_tags": train_en_labels}})
    column_names = raw_datasets["train_en"].column_names
    features = raw_datasets["train_en"].features

    for key in list(raw_datasets.keys()):
        if key != "train_en":
            del raw_datasets[key]

    text_column_name = "new_tokens"
    label_column_name = "ner_tags"
    lang_column_name = "lang"

    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.
    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train_en"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}

    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, add_prefix_space=True)
    # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True, add_prefix_space=True)
    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label >= 1 and label % 2 == 1:
            b_to_i_label.append(idx + 1)
        else:
            b_to_i_label.append(idx)

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
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])

                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["ner_labels"] = labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        keep_in_memory=True,
        remove_columns=raw_datasets["train_en"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_raw_datasets["train_en"]
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=100
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
    optimizer = AdamW(model.parameters(), LR_PRETRAIN)
    scheduler = ReverseSqrtScheduler(optimizer, [LR_PRETRAIN], WARMUP_STEPS)

    epoch_start = 0

    for epoch in range(epoch_start, 100):
        model.train()
        all_loss = 0
        update_step = 0
        train_dataloader = get_train_dataloader()
        for batch in tqdm(train_dataloader):
            loss, _ = model(**batch)
            all_loss += loss.item()
            loss.backward()
            scheduler.step_and_update_lr()
            scheduler.zero_grad()
            update_step += 1
            # if update_step % 100 == 0:
            #     print("epoch: {}, Update Steps {}, loss: {}\n".format(epoch, update_step, all_loss / update_step))
            # f update_step % 1000 == 0:
            #    print("epoch: {}, Update Steps {}, loss: {}\n".format(epoch, update_step, all_loss / update_step))
            #    with torch.no_grad():
            #        model.eval()
            #        evaluate(model, eval_dataloaders)

            # model.train()
        print("epoch: {}, Update Steps {}, loss: {}\n".format(epoch, update_step, all_loss / update_step))
        with torch.no_grad():
            model.eval()
            evaluate(model, eval_dataloaders)
        # s.system("rm data/huggingface/parquet/Babelscape--wikineural-2c05bc228ce4015c/0.0.0/0b6d5799bb726b24ad7fc7be720c170d8e497f575d02d47537de9a5bac074901/*")


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
        fw.write(str(res) + '\n')

    print(res)


def evaluate(model, dataloaders):
    for lg, dataloader in dataloaders.items():
        with open("results", 'a') as fw:
            fw.write(lg + '\n')
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