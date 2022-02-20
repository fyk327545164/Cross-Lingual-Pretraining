# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--model-name', required=True, type=str)
import numpy as np
from transformers import AutoTokenizer, AdamW, default_data_collator, XLMRobertaModel

from torch.nn.modules.loss import BCEWithLogitsLoss
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import torch.utils.data.distributed
import random

import os
# os.system("wget -O ende.zip https://opus.nlpl.eu/download.php?f=News-Commentary/v16/moses/de-en.txt.zip")
# os.system("unzip ende.zip")


lg = ["en", "de"]

LR_PRETRAIN = 0.00006
LR_FINETUNE = 0.00009
WARMUP_STEPS = 500


class CrossLingualModel(nn.Module):
    def __init__(self, num_lg=2):
        super().__init__()

        self.xlm = XLMRobertaModel.from_pretrained("xlm-roberta-base")

        self.align_embedding = self.xlm.embeddings.word_embeddings.weight.clone()  # V * D

        print(self.align_embedding.size())

        self.hidden_size = 768

        self.lg_projections = {}
        for i in range(num_lg):
            for j in range(num_lg):
                self.lg_projections["{}-{}".format(i, j)] = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(0.1)

        # self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.loss = BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, lg_in, lg_out, labels_src, labels_tgt):
        input_ids = input_ids.to(self.xlm.device)
        attention_mask = attention_mask.to(self.xlm.device)

        outputs = self.xlm(input_ids, attention_mask)

        sequence_output = self.dropout(outputs[0])

        logits_src = self.lg_projections["{}-{}".format(lg_in, lg_in)](sequence_output)
        logits_tgt = self.lg_projections["{}-{}".format(lg_in, lg_out)](sequence_output)

        logits_src = torch.matmul(logits_src, self.align_embedding.T)  # T * V
        logits_tgt = torch.matmul(logits_tgt, self.align_embedding.T)  # T * V

        logits_src = logits_src.transpose(1,2)  # V * T
        logits_tgt = logits_tgt.transpose(1,2)  # V * T

        logits_src = torch.sum(logits_src * nn.Softmax(dim=-1)(logits_src), -1)
        logits_tgt = torch.sum(logits_tgt * nn.Softmax(dim=-1)(logits_tgt), -1)

        loss = None
        if labels_src is not None:
            labels_src = labels_src.to(self.xlm.device)
            labels_tgt = labels_tgt.to(self.xlm.device)

            # print(logits_src.size(), labels_src.size())

            loss_src = self.loss(logits_src.view(-1), labels_src.view(-1).float())
            loss_tgt = self.loss(logits_tgt.view(-1), labels_tgt.view(-1).float())

            loss = loss_src + loss_tgt

        return loss, (logits_src, logits_tgt)


def process_task_data(tokenizer):
    max_length = 128

    filename_src = "News-Commentary.de-en.en"
    filename_tgt = "News-Commentary.de-en.de"

    datasets = []
    with open(filename_src, 'r') as fr_src, open(filename_tgt, 'r') as fr_tgt:
        src_lines = fr_src.readlines()
        tgt_lines = fr_tgt.readlines()
        assert len(src_lines) == len(tgt_lines)
        for i in trange(len(src_lines[:10000])):
            src_line = src_lines[i][:-1]
            tgt_line = tgt_lines[i][:-1]

            if len(src_line.split()) == 0:
                continue

            src_tokens = tokenizer.tokenize(src_line)
            tgt_tokens = tokenizer.tokenize(tgt_line)

            src_input_ids = tokenizer.convert_tokens_to_ids(src_tokens)
            tgt_input_ids = tokenizer.convert_tokens_to_ids(tgt_tokens)

            src_attention_mask = [1 for _ in range(len(src_input_ids))]
            tgt_attention_mask = [1 for _ in range(len(tgt_input_ids))]

            while len(src_input_ids) < max_length:
                src_input_ids.append(tokenizer.pad_token_id)
                src_attention_mask.append(0)
            src_input_ids = src_input_ids[:max_length]
            src_attention_mask = src_attention_mask[:max_length]

            while len(tgt_input_ids) < max_length:
                tgt_input_ids.append(tokenizer.pad_token_id)
                tgt_attention_mask.append(0)
            tgt_input_ids = tgt_input_ids[:max_length]
            tgt_attention_mask = tgt_attention_mask[:max_length]

            datasets.append({"src_input_ids": np.array(src_input_ids),
                             "src_attention_mask": np.array(src_attention_mask),
                             "tgt_input_ids": np.array(tgt_input_ids),
                             "tgt_attention_mask": np.array(tgt_attention_mask)})

    return datasets


class AlignDataset(Dataset):
    def __init__(self, ds):
        self.dataset = ds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


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
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    print(tokenizer.vocab_size)
    model = CrossLingualModel()

    train_dataset = AlignDataset(process_task_data(tokenizer))
    train_dataloader = DataLoader(dataset=train_dataset, pin_memory=True, batch_size=8, shuffle=True)

    # model.cuda()

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

    epoch_start = 0

    _ids = [0 for _ in range(tokenizer.vocab_size)]
    for epoch in range(epoch_start, 100):
        model.train()
        all_loss = 0
        update_step = 0

        for batch in tqdm(train_dataloader):
            src_input_ids = batch["src_input_ids"]
            src_attention_mask = batch["src_attention_mask"]
            tgt_input_ids = batch["tgt_input_ids"]
            tgt_attention_mask = batch["tgt_attention_mask"]

            # src_input_ids_list = src_input_ids.tolist()
            # tgt_input_ids_list = tgt_input_ids.tolist()

            if random.choice(lg) == "en":
                lg_in = 0
                lg_out = 1
                input_ids, attention_mask = src_input_ids, src_attention_mask

                labels_src = torch.zeros((src_input_ids.size()[0], tokenizer.vocab_size))
                labels_tgt = torch.zeros((tgt_input_ids.size()[0], tokenizer.vocab_size))

                labels_src[torch.arange(labels_src.size(0)).unsqueeze(1), src_input_ids.masked_fill(tgt_input_ids == -100, tokenizer.eos_token_id)] = 1
                labels_tgt[torch.arange(labels_tgt.size(0)).unsqueeze(1), tgt_input_ids.masked_fill(tgt_input_ids == -100, tokenizer.eos_token_id)] = 1

                # for src_label_list in src_input_ids_list:
                #     cur_src_label = [0 for _ in range(tokenizer.vocab_size)]
                #     for _id in set(src_label_list):
                #         cur_src_label[_id] = 1
                #     labels_src.append(cur_src_label)

                # for tgt_label_list in tgt_input_ids_list:
                #     cur_tgt_label = [0 for _ in range(tokenizer.vocab_size)]
                #     for _id in set(tgt_label_list):
                #         cur_tgt_label[_id] = 1
                #     labels_tgt.append(cur_tgt_label)

            else:
                lg_in = 1
                lg_out = 0
                input_ids, attention_mask = tgt_input_ids, tgt_attention_mask

                labels_src = torch.zeros((src_input_ids.size()[0], tokenizer.vocab_size))
                labels_tgt = torch.zeros((tgt_input_ids.size()[0], tokenizer.vocab_size))

                labels_src[torch.arange(labels_src.size(0)).unsqueeze(1), tgt_input_ids.masked_fill(tgt_input_ids == -100, tokenizer.eos_token_id)] = 1
                labels_tgt[torch.arange(labels_tgt.size(0)).unsqueeze(1), src_input_ids.masked_fill(tgt_input_ids == -100, tokenizer.eos_token_id)] = 1

            labels_src[:, tokenizer.eos_token_id] = 0
            labels_tgt[:, tokenizer.eos_token_id] = 0

            # labels_src = torch.Tensor(labels_src)
            # labels_tgt = torch.Tensor(labels_tgt)

            loss, _ = model(input_ids, attention_mask, lg_in, lg_out, labels_src, labels_tgt)
            all_loss += loss.item()
            loss.backward()
            scheduler.step_and_update_lr()
            scheduler.zero_grad()
            update_step += 1
            if update_step % 100 == 0:
                print("epoch: {}, Update Steps {}, loss: {}\n".format(epoch, update_step, all_loss / update_step))

        print("epoch: {}, Update Steps {}, loss: {}\n".format(epoch, update_step, all_loss / update_step))

        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler": scheduler},
                   "ckpt-{}".format(epoch))


main()
