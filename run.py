import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', required=True, type=str)
parser.add_argument('--task-name', required=True, type=str)
parser.add_argument("--use-context", required=True, type=int)
parser.add_argument("--segment-length", required=True, type=int)
parser.add_argument("--context-length", required=True, type=int)
parser.add_argument("--ckpt-index", required=True, type=int)
args = parser.parse_args()

CKPT_INDEX = args.ckpt_index
TASK_NAME = args.task_name
MODEL_NAME = args.model_name
SEGMENT_LENGTH = args.segment_length
USE_CONTEXT = True if args.use_context == 1 else False
CONTEXT_LENGTH = args.context_length

WARMUP_STEPS = 1000
LR = 6e-5#- 1e-5*CKPT_INDEX

from transformers import AutoTokenizer, AdamW, default_data_collator

import sys
import json
sys.path.append('../')

from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import random
import pickle
import numpy as np
import os
from datasets import load_metric, ClassLabel

import spacy

import torch.utils.data.distributed


from datasets import load_dataset


nlp = spacy.load("en_core_web_sm")
metric = load_metric("seqeval")


class CrossLingualModel(nn.Module):
    def __init__(self, num_labels=8):
        super().__init__()

        self.model = Seq2Seq.get_enecoder_model(args.model_name, use_context=True if args.use_context == 1 else False,
                                             segment_length=args.segment_length,
                                             context_length=args.context_length,
                                             use_global=False)
        self.hidden_size = 768
        self.num_labels = num_labels

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.loss = CrossEntropyLoss()

    def forward(self, input_ids, label_ids=None):
        attention_mask = torch.ones(input_ids.size()).cuda().long()

        outputs = self.model(input_ids, mask=attention_mask)

        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)

        loss = None
        if label_ids is not None:
            loss = self.loss(logits.view(-1, self.num_labels), label_ids.view(-1))

        return loss, logits


class NERDataset(Dataset):
    def __init__(self, ds):
        self.dataset = ds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


LANGUAGE_IDS = ["de", "en", "es", "nl", "pl", "ru", "it", "fr", "pt"]
# https://github.com/Babelscape/wikineural
def process_task_data():
    """


    :return: {
        "train": NERDataset,
        "test": {
            "lg": NERDataset,
            ....
            }
        }
    """
    raw_datasets = load_dataset(args.dataset_name)

    column_names = raw_datasets["train_en"].column_names
    features = raw_datasets["train_en"].features

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

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True, add_prefix_space=True)

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
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
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets["train_en"].column_names,
            desc="Running tokenizer on dataset",
    )

    train_dataset = processed_raw_datasets["train"]
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=2
    )

    eval_dataloaders = {}
    for lg in LANGUAGE_IDS:
        eval_dataloaders[lg] = DataLoader(processed_raw_datasets["val_{}".format(lg)], collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)

    return train_dataloader, eval_dataloaders


def process_parallel_data():

    return








def process_data(tokenizer, mode="train"):

    if os.path.exists("{}-cache-{}".format(mode, TASK_NAME)):

        with open("{}-cache-{}".format(mode, TASK_NAME), 'rb') as fr:
            docs = pickle.load(fr)
    else:

        label_to_id = {"MV": 1, "GP": 2, "FP": 3, "BG": 4, "PL": 5, "PC": 6, "HS": 7}
        # Music Video: [MV-B], [MV-I]
        # Game Play: [GP-B], [GP-I]
        # Film Plot: [FP-B], [FP-I]
        # Building Geography: [BG-B], [BG-I]
        # Person Life: [PL-B], [PL-I]
        # Person Career: [PC-B], [PC-I]
        # Other history: [HIS-B], [HIS-I]

        docs = []
        with open("{}-text".format(mode), 'r', encoding='utf-8') as fr:
            lines = fr.readlines()

        for line in tqdm(lines):
            sents = nlp(line)
            doc_input_ids = []
            doc_label_ids = []

            sent_label = 0
            breaked = False
            for sent_str in list(sents.sents):
                sent = str(sent_str)

                if "-END-TAG]" in sent:
                    sent_label = 0
                    label = sent[sent.index('-END-TAG]') - 2:sent.index('-END-TAG]')]
                    # if label  in label_to_id:
                    # else:
                    #     label = sent[sent.index('-END]')-3:sent.index('-END]')]
                    #     sent_label = label_to_id[label]
                    full_label = "[" + label + "-END-TAG]"
                    sent = sent.replace(full_label, "")

                if "-BEGIN-TAG]" in sent:
                    try:
                        assert sent_label == 0
                    except:
                        breaked = True
                        print("breaked")
                        break
                    label = sent[sent.index('-BEGIN-TAG]') - 2:sent.index('-BEGIN-TAG]')]
                    # if label  in label_to_id:
                    sent_label = label_to_id[label]
                    # else:
                    #     label = sent[sent.index('-BEGIN-TAG]')-3:sent.index('-BEGIN-TAG]')]
                    #     sent_label = label_to_id[label]
                    full_label = "[" + label + "-BEGIN-TAG]"
                    sent = sent.replace(full_label, "")

                # print(sent_label)

                cur_tokens = tokenizer.tokenize(sent)
                cur_tokens = [tokenizer.cls_token] + cur_tokens[:] + [tokenizer.sep_token]

                input_ids = tokenizer.convert_tokens_to_ids(cur_tokens)
                label_ids = [-100 for _ in range(len(input_ids))]
                label_ids[0] = sent_label

                doc_input_ids += input_ids
                doc_label_ids += label_ids
            if breaked:
                continue
            docs.append([doc_input_ids, doc_label_ids])

        with open("{}-cache-{}".format(mode, TASK_NAME), 'wb') as fw:
            pickle.dump(docs, fw)
    return docs#:100]
    pad_index = tokenizer.pad_token_id
    new_docs = []
    for doc in docs:
        doc_input_ids, doc_label_ids = doc
        if len(doc_input_ids) % 2 == 0:
            doc_input_ids.append(pad_index)
            doc_label_ids.append(-100)
        new_docs.append([doc_input_ids, doc_label_ids])
    return new_docs


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
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    train_dataloader = DataLoader(dataset=RegionDetectionDataset(process_data(tokenizer, "train")), pin_memory=True,
                                  batch_size=1, shuffle=True)
    test_dataloader = DataLoader(dataset=RegionDetectionDataset(process_data(tokenizer, "test")), pin_memory=True,
                                 batch_size=1, shuffle=True)

    model = RegionDetectionModel()
    model.cuda()
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = ReverseSqrtScheduler(optimizer, [LR], WARMUP_STEPS)

    epoch_start = 0
    if CKPT_INDEX != 0:
        checkpoint = torch.load("ckpt-{}-{}-{}".format(TASK_NAME, USE_CONTEXT, CKPT_INDEX-1))
        epoch_start = checkpoint["epoch"]+1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _scheduler = checkpoint["scheduler"]
        scheduler.n_steps = _scheduler.n_steps
    #optimizer = AdamW(model.parameters(), lr=LR)
    #scheduler = ReverseSqrtScheduler(optimizer, [LR], WARMUP_STEPS)

    with torch.no_grad():
        model.eval()
        evaluate(model, test_dataloader)

    for epoch in range(epoch_start, 100):
        model.train()
        all_loss = 0
        update_step = 0

        for batch in tqdm(train_dataloader):
            input_ids, label_ids = batch
            input_ids = torch.Tensor([input_ids]).cuda().long()
            label_ids = torch.Tensor([label_ids]).cuda().long()
            loss, _ = model(input_ids, label_ids)
            all_loss += loss.item()
            loss.backward()
            scheduler.step_and_update_lr()
            scheduler.zero_grad()
            update_step += 1
            #if update_step % 100 == 0:
            #    print("Update Steps {} loss: {}\n".format(update_step, all_loss / update_step))

        print("epoch: {}, Update Steps {}, loss: {}\n".format(epoch, update_step, all_loss / update_step))
        with torch.no_grad():
            model.eval()
            evaluate(model, test_dataloader)
        torch.save({"epoch":epoch, "model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(), "scheduler":scheduler}, "ckpt-{}-{}-{}".format(TASK_NAME, USE_CONTEXT, CKPT_INDEX))


def compute_metrics(p):
    predictions, labels = p
    label_list = ["O", "MV", "GP", "FP", "BG", "PL", "PC", "HS"]
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[int(p)] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[int(l)] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    with open("results", 'w') as fw:
        for p, t in zip(true_predictions, true_labels):
            fw.write("{} {} \n".format(p, t))
    results = metric.compute(predictions=true_predictions, references=true_labels)

    print({"precision": results["overall_precision"],
           "recall": results["overall_recall"],
           "f1": results["overall_f1"],
           "accuracy": results["overall_accuracy"]})


def evaluate(model, dataloader):
    preds = []
    targets = []

    for batch in tqdm(dataloader):
        input_ids, label_ids  = batch
        #batch = torch.transpose(batch, 0, 2)
        #input_ids, attention_mask, label_ids = batch
        input_ids = torch.Tensor([input_ids]).to("cuda").long()
        #print(input_ids.size())
        label_ids = torch.Tensor([label_ids])
        #attention_mask = attention_mask.to("cuda").long()
        label_ids = label_ids.view(-1, 1).tolist()
        _, logits = model(input_ids)
        logits = logits.cpu()
        #rint(logits.size())
        index = torch.argmax(logits, -1).view(-1, 1).tolist()
        #print(len(index), len(label_ids))
        #assert len(index) == len(label_ids)
        preds += index[:]
        targets += label_ids[:]
    # print(preds, targets)
    compute_metrics((preds, targets))

main()