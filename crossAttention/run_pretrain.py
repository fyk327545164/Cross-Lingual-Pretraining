import collections
import math
import os.path
import random

import numpy as np
import torch
from tqdm import trange, tqdm
import argparse
from torch.utils.data import DataLoader, Dataset
from transformers import BartConfig, AdamW, AutoTokenizer

from EncoderDecoder import EncoderDecoderModel

parser = argparse.ArgumentParser()
parser.add_argument('--cross-attention-index', required=True, type=str)
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--layer', required=True, type=int)
parser.add_argument('--batch-size', required=True, type=int)
parser.add_argument('--seed', required=True, type=int)

args = parser.parse_args()

cross_attention_index = [None if c == "0" else int(c) for c in args.cross_attention_index.split('.')]
BATCH_SIZE = args.batch_size
DATASET = args.dataset
LAYER = args.layer
SEED = args.seed
print(cross_attention_index)

LR = 0.00008
WARMUP_STEPS = 1000

DROPOUT=0.3 if "wmt" not in DATASET else 0.1

fw = open("res-{}-{}-{}-{}".format(DATASET,LAYER,"".join(args.cross_attention_index.split('.')), SEED), 'w')


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.
    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return bleu, precisions, bp, ratio, translation_length, reference_length


def process_data(filename_src, filename_tgt, encoder_tokenizer, decoder_tokenizer):
    if os.path.exists("{}-cache".format(filename_src)):
        with open("{}-cache".format(filename_src), 'rb') as fw:
            import pickle
            datasets = pickle.load(fw)
        return datasets

    max_length = 100

    datasets = []

    with open(filename_src, 'r') as fr_src, open(filename_tgt, 'r') as fr_tgt:
        src_lines = fr_src.readlines()
        tgt_lines = fr_tgt.readlines()
        assert len(src_lines) == len(tgt_lines)

        if "valid" in filename_src:
            nums = min(len(src_lines[:]), 1000)
        else:
            nums = len(src_lines[:])

        for i in trange(nums):
            src_line = src_lines[i][:-1]
            tgt_line = tgt_lines[i][:-1]

            if len(src_line.split()) == 0:
                continue
            src_tokens = encoder_tokenizer.tokenize(src_line.replace("@@ ", ""))
            tgt_tokens = decoder_tokenizer.tokenize(tgt_line.replace("@@ ", ""))

            src_input_ids = encoder_tokenizer.convert_tokens_to_ids(src_tokens)
            tgt_input_ids = [50265] + decoder_tokenizer.convert_tokens_to_ids(tgt_tokens) + [50266]
            label_ids = tgt_input_ids[1:]
            tgt_input_ids = tgt_input_ids[:-1]

            src_attention_mask = [1 for _ in range(len(src_input_ids))]
            tgt_attention_mask = [1 for _ in range(len(tgt_input_ids))]

            if len(src_input_ids) > max_length or len(tgt_input_ids) > max_length:
                continue

            while len(src_input_ids) < max_length:
                src_input_ids.append(0)
                src_attention_mask.append(0)

            while len(label_ids) < max_length:
                tgt_input_ids.append(0)
                tgt_attention_mask.append(0)
                label_ids.append(-100)

            datasets.append({"src_input_ids": np.array(src_input_ids),
                             "src_attention_mask": np.array(src_attention_mask),
                             "tgt_input_ids": np.array(tgt_input_ids),
                             "tgt_attention_mask": np.array(tgt_attention_mask),
                             "tgt_label_ids": np.array(label_ids)})

    with open("{}-cache".format(filename_src), 'wb') as fw:
        import pickle
        pickle.dump(datasets, fw)

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
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    encoder_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    decoder_tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")

    ckpt_path = "ckpt-{}-{}-{}-{}".format(DATASET,LAYER,"".join(args.cross_attention_index.split('.')),SEED)

    train_dataset = AlignDataset(process_data("../data/{}/train.en".format(DATASET), "../data/{}/train.de".format(DATASET), encoder_tokenizer, decoder_tokenizer))
    dev_dataset = AlignDataset(process_data("../data/{}/valid.en".format(DATASET), "../data/{}/valid.de".format(DATASET), encoder_tokenizer, decoder_tokenizer))
    test_dataset = AlignDataset(process_data("../data/{}/test.en".format(DATASET), "../data/{}/test.de".format(DATASET), encoder_tokenizer, decoder_tokenizer))

    train_dataloader = DataLoader(dataset=train_dataset, pin_memory=True, batch_size=BATCH_SIZE, shuffle=True)
    dev_dataloader = DataLoader(dataset=dev_dataset, pin_memory=True, batch_size=BATCH_SIZE)#//2)# shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, pin_memory=True, batch_size=BATCH_SIZE)#//2)# shuffle=True)

    # config = BartConfig(vocab_size=len(vocab_to_ids), d_model=512, max_position_embeddings=105,
    #                     encoder_ffn_dim=2048, decoder_ffn_dim=2048,
    #                     pad_token_id=0, bos_token_id=1, eos_token_id=2,
    #                     decoder_start_token_id=1, encoder_layers=LAYER,
    #                     decoder_layers=LAYER, encoder_attention_heads=8, decoder_attention_heads=8, dropout=DROPOUT,#0.3,
    #                     attention_dropout=DROPOUT,
    #                     cross_attention_index=cross_attention_index
    #                     )

    # model = EncoderDecoderModel(cross_attention_index=cross_attention_index)
    model = EncoderDecoderModel(cross_attention_index=cross_attention_index)
    model.cuda()

    optimizer = AdamW(model.parameters(), LR, (0.9, 0.98))
    scheduler = ReverseSqrtScheduler(optimizer, [LR], WARMUP_STEPS)

    epoch_start = 0
    test_result = []
    dev_result = []
    # if os.path.exists(ckpt_path):
    #     checkpoint = torch.load(ckpt_path)
    #     epoch_start = checkpoint["epoch"] + 1
    #     model.load_state_dict(checkpoint["model_state_dict"])
    #     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #     _scheduler = checkpoint["scheduler"]
    #     test_result = checkpoint["test-result"]
    #     dev_result = checkpoint["dev-result"]
    #     layer_entropy = checkpoint["layer_entropy"]
    #     for i, (self_entropy, cross_entropy) in enumerate(layer_entropy):
    #         model.model.decoder.layers[i].cross_entropy = cross_entropy[:]
    #         model.model.decoder.layers[i].self_entropy = self_entropy[:]
    #     del layer_entropy
    #     scheduler.n_steps = _scheduler.n_steps
    #     del _scheduler
    #     del checkpoint

    for epoch in range(epoch_start, 100):
        model.train()
        all_loss = 0
        update_step = 0

        for batch in tqdm(train_dataloader):
            src_input_ids = batch["src_input_ids"].to(model.device)
            src_attention_mask = batch["src_attention_mask"].to(model.device)
            tgt_input_ids = batch["tgt_input_ids"].to(model.device)
            tgt_attention_mask = batch["tgt_attention_mask"].to(model.device)
            tgt_label_ids = batch["tgt_label_ids"].to(model.device)
            loss = model(src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask, labels=tgt_label_ids)[
                "loss"]
            all_loss += loss.item()
            loss.backward()
            scheduler.step_and_update_lr()
            scheduler.zero_grad()
            update_step += 1
            if update_step % 1000 == 0:
                print("epoch: {}, Update Steps {}, loss: {}".format(epoch, update_step, all_loss / update_step))
                if update_step % 5000 == 0:
                    dev_res = evaluate(model, dev_dataloader, decoder_tokenizer)
                    test_res = evaluate(model, test_dataloader, decoder_tokenizer)

                    # layer_entropy = [(layer.self_entropy, layer.cross_entropy) for layer in model.model.decoder.layers]
                    dev_result.append(dev_res)
                    test_result.append(test_res)
                    # torch.save(
                    #     {"epoch": epoch, "model_state_dict": model.state_dict(),
                    #      "optimizer_state_dict": optimizer.state_dict(),
                    #      "scheduler": scheduler, "dev-result": dev_result, "test-result": test_result,
                    #      "layer_entropy": layer_entropy}, ckpt_path)

                    best_res = max(dev_result)
                    best_index = dev_result.index(best_res)
                    max_index = len(dev_result) - 1

                    if (max_index - best_index) > 5:
                        print("dev: {}".format(str(dev_result)))
                        print("test: {}".format(str(test_result)))
                        print("final_dev: {}, final_test: {}".format(dev_result[best_index], test_result[best_index]))
                        exit(-1)

        print("epoch: {}, Update Steps {}, loss: {}".format(epoch, update_step, all_loss / update_step))
        dev_res = evaluate(model, dev_dataloader, decoder_tokenizer)
        test_res = evaluate(model, test_dataloader, decoder_tokenizer)
        # layer_entropy = [(layer.self_entropy, layer.cross_entropy) for layer in model.model.decoder.layers]

        dev_result.append(dev_res)
        test_result.append(test_res)
        # torch.save(
        #     {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        #      "scheduler": scheduler, "dev-result":dev_result, "test-result":test_result, "layer_entropy": layer_entropy}, ckpt_path)

        best_res = max(dev_result)
        best_index = dev_result.index(best_res)
        max_index = len(dev_result) - 1

        if (max_index - best_index) > 5:
            print("dev: {}".format(str(dev_result)))
            print("test: {}".format(str(test_result)))
            print("final_dev: {}, final_test: {}".format(dev_result[best_index], test_result[best_index]))
            exit(-1)


def evaluate(model, test_dataloader, tokenizer):
    eos = 50266 #tokenizer.eos_token_id
    sos = 50265 #tokenizer.bos_token_id

    preds = []
    targets = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_dataloader):
            src_input_ids = batch["src_input_ids"].cuda()
            src_attention_mask = batch["src_attention_mask"].cuda()
            tgt_input_ids = torch.zeros((src_input_ids.size()[0], 1)).cuda().long() + sos
            tgt_label_ids = batch["tgt_label_ids"].tolist()

            outputs = model.generate(input_ids=src_input_ids,
                                     attention_mask=src_attention_mask,
                                     decoder_input_ids=tgt_input_ids,
                                     bos_token_id=sos,
                                     num_return_sequences=1,
                                     num_beams=5,
                                     max_length=100)# use_cache=False)
            outputs = outputs.cpu().tolist()
            assert len(outputs) == len(tgt_label_ids)

            for i in range(len(outputs)):
                pred = outputs[i]
                if eos in pred:
                    index = pred.index(eos)
                    pred = pred[:index]
                pred_str = tokenizer.decode(pred[1:])

                target = tgt_label_ids[i]
                index = target.index(eos)
                target_str = tokenizer.decode(target[:index])

                preds.append(pred_str.split())
                targets.append([target_str.split()])

    model.train()
    score = float(compute_bleu(targets, preds, max_order=4)[0])
    # print("BLEU score: {}".format(score))
    return score

main()
