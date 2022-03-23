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
from collections import Counter

np.random.seed(1234)
torch.manual_seed(1234)
random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', required=True, type=int)
parser.add_argument('--lg', required=True, type=str)
args = parser.parse_args()


LR_PRETRAIN = 0.00002
LR_FINETUNE = 0.00008
WARMUP_STEPS = 200

MODEL_NAME = "bert-base-multilingual-cased"

with open("data/aligned-tokens-en-{}".format(args.lg), "rb") as fr:
    aligned_tokens = pickle.load(fr)
"""
aligned_tokens = {}
for key, val in _aligned_tokens.items():
    aligned_tokens[key] = []
    for _word, _count in Counter(val).most_common():
        if len(aligned_tokens[key]) == 0:
            aligned_tokens[key].append(_word)
            #reak
            _all_count = _count
            continue
        if _count/_all_count < 0.88:
            break
        aligned_tokens[key].append(_word)  
"""
random_index = [0 for _ in range(args.ratio)] + [1 for _ in range(100 - args.ratio)]
random.shuffle(random_index)
print(random_index)

res_path = "res/xnli-{}-{}".format(args.lg, args.ratio)


class CrossLingualModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.xlm = BertModel.from_pretrained(MODEL_NAME)

        self.hidden_size = 768
        self.dropout = nn.Dropout(0.1)

        self.num_labels = 3
        self.nli_classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.loss = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        input_ids = input_ids.to(self.xlm.device)
        attention_mask = attention_mask.to(self.xlm.device)
        outputs = self.xlm(input_ids, attention_mask)

        sequence_output = self.dropout(outputs[1])
        logits = self.nli_classifier(sequence_output)

        loss = None
        if labels is not None:
            ner_labels = labels.to(self.xlm.device)
            loss = self.loss(logits.view(-1, self.num_labels), ner_labels.view(-1))

        return loss, logits


LANGUAGE_IDS = [args.lg]


def process_task_data():

    eval_dataset = load_dataset("xnli", LANGUAGE_IDS[0], split="test")  # ["train_en", "val_en"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_function(examples):
        # Tokenize the texts
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            max_length=128,
            truncation=True,
        )

    eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=True,
                desc="Running tokenizer on validation dataset",
            )

    eval_dataloaders = {}
    for lg in LANGUAGE_IDS:
        eval_dataloaders[lg] = DataLoader(eval_dataset, collate_fn=default_data_collator,
                                          batch_size=64)

    return eval_dataloaders


def get_train_dataloader():

    train_dataset = load_dataset("xnli", "en", split="train")

    train_premise = train_dataset["premise"][:]
    train_hypothesis = train_dataset["hypothesis"][:]

    train_dataset = train_dataset.remove_columns(["premise", "hypothesis"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    new_premise = []
    new_hypothesis = []
    for i in trange(len(train_premise)):
        tokenized_premise = " ".join(tokenizer.tokenize(train_premise[i])).replace(" ##", "").split()
        tokenized_hypothesis = " ".join(tokenizer.tokenize(train_hypothesis[i])).replace(" ##", "").split()

        for token_i in range(len(tokenized_premise)):
            cur_token = tokenized_premise[token_i].lower()
            if cur_token in aligned_tokens:
                if random.choice(random_index) == 0:
                    tokenized_premise[token_i] = random.choice(aligned_tokens[cur_token])

        for token_i in range(len(tokenized_hypothesis)):
            cur_token = tokenized_hypothesis[token_i].lower()
            if cur_token in aligned_tokens:
                if random.choice(random_index) == 0:
                    tokenized_hypothesis[token_i] = random.choice(aligned_tokens[cur_token])
        new_premise.append(" ".join(tokenized_premise))
        new_hypothesis.append(" ".join(tokenized_hypothesis))

    train_dataset = train_dataset.add_column("premise", new_premise)
    train_dataset = train_dataset.add_column("hypothesis", new_hypothesis)


    def preprocess_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            max_length=128,
            truncation=True,
        )

    train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=False,
    )

    train_dataloader = DataLoader(train_dataset, collate_fn=default_data_collator, shuffle=True,
                                          batch_size=64)

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
    scheduler = ReverseSqrtScheduler(optimizer, [LR_PRETRAIN, LR_FINETUNE], WARMUP_STEPS)
    # optimizer = AdamW(model.parameters(), LR_PRETRAIN)
    # scheduler = ReverseSqrtScheduler(optimizer, [LR_PRETRAIN], WARMUP_STEPS)

    # with open("xnli-results", 'a') as fw:
    #     fw.write('----------------------\n')

    for epoch in range(6):
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

        print("epoch: {}, Update Steps {}, loss: {}\n".format(epoch, update_step, all_loss / update_step))
        with torch.no_grad():
            model.eval()
            evaluate(model, eval_dataloaders)


def evaluate(model, dataloaders):
    for lg, dataloader in dataloaders.items():
        # with open(res_path, 'a') as fw:
        #     fw.write('{}-{} \n'.format(args.lg, args.ratio))
        _evaluate(model, dataloader, lg)


def _evaluate(model, dataloader, lg):
    acc = 0
    nums = 0

    for batch in tqdm(dataloader):
        label_ids = batch["labels"].view(-1).tolist()

        _, logits = model(batch["input_ids"], batch["attention_mask"])

        logits = logits.cpu()
        preds = torch.argmax(logits, -1).view(-1).tolist()

        for p, label in zip(preds, label_ids):
            nums += 1
            if p == label:
                acc += 1

    print("Accuracy: {}".format(acc/nums))

    with open(res_path, 'a') as fw:
        fw.write("Accuracy: {}\n".format(acc/nums))


if __name__ == "__main__":
    main()