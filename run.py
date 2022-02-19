# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--model-name', required=True, type=str)

from transformers import AutoTokenizer, AdamW, default_data_collator, XLMRobertaModel

from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from datasets import load_metric, ClassLabel
import torch.utils.data.distributed

from datasets import load_dataset

metric = load_metric("seqeval")

global_label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']


LR_PRETRAIN = 0.00001
LR_FINETUNE = 0.00006
WARMUP_STEPS = 500

class CrossLingualModel(nn.Module):
    def __init__(self, num_labels=len(global_label_list)):
        super().__init__()

        self.xlm = XLMRobertaModel.from_pretrained("xlm-roberta-base")

        self.hidden_size = 768
        self.num_labels = num_labels

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.loss = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, ner_labels=None):
        input_ids = input_ids.to(self.xlm.device)
        attention_mask = attention_mask.to(self.xlm.device)

        outputs = self.xlm(input_ids, mask=attention_mask)

        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)

        loss = None
        if ner_labels is not None:
            ner_labels = ner_labels.to(self.xlm.device)
            loss = self.loss(logits.view(-1, self.num_labels), ner_labels.view(-1))

        return loss, logits


LANGUAGE_IDS = ["en", "de", "en", "es", "nl", "pl", "ru", "it", "fr", "pt"]


# https://github.com/Babelscape/wikineural
def process_task_data():
    raw_datasets = load_dataset("Babelscape/wikineural")  # ["train_en", "val_en"])

    column_names = raw_datasets["train_en"].column_names
    features = raw_datasets["train_en"].features

    for key in list(raw_datasets.keys()):
        if "train" in key and key != "train_en":
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

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True, add_prefix_space=True)

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

    train_dataset = processed_raw_datasets["train_en"]
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=2
    )

    eval_dataloaders = {}
    for lg in LANGUAGE_IDS:
        eval_dataloaders[lg] = DataLoader(processed_raw_datasets["val_{}".format(lg)], collate_fn=default_data_collator,
                                          batch_size=2)

    return train_dataloader, eval_dataloaders


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
    train_dataloader, eval_dataloaders = process_task_data()

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

    epoch_start = 0

    for epoch in range(epoch_start, 100):
        model.train()
        all_loss = 0
        update_step = 0

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

        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler": scheduler},
                   "ckpt-{}".format(epoch))


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

    with open("results", 'a') as fw:
        for p, t in zip(true_predictions, true_labels):
            fw.write("{} {} \n".format(p, t))
    results = metric.compute(predictions=true_predictions, references=true_labels)

    print({"precision": results["overall_precision"],
           "recall": results["overall_recall"],
           "f1": results["overall_f1"],
           "accuracy": results["overall_accuracy"]})


def evaluate(model, dataloaders):
    for lg, dataloader in dataloaders.items():
        with open("results", 'a') as fw:
            fw.write(lg+'\n')
        print(lg)
        _evaluate(model, dataloader)


def _evaluate(model, dataloader):
    preds = []
    targets = []

    for batch in tqdm(dataloader):
        input_ids, attention_mask, label_ids = batch
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        label_ids = label_ids.view(-1, 1).tolist()

        _, logits = model(input_ids, attention_mask)
        logits = logits.cpu()
        index = torch.argmax(logits, -1).view(-1, 1).tolist()

        preds += index[:]
        targets += label_ids[:]

    compute_metrics((preds, targets))


main()
