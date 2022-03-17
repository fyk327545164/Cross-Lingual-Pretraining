from tqdm import trange
from transformers import AutoTokenizer
import pickle


def create_parallel_data():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    filename_src = "data/News-Commentary.en-ru.en"
    filename_tgt = "data/News-Commentary.en-ru.ru"

    with open(filename_src, 'r') as fr_src, open(filename_tgt, 'r') as fr_tgt, open("data/parallel_sentences.txt",
                                                                                    "w") as fw:
        src_lines = fr_src.readlines()
        tgt_lines = fr_tgt.readlines()
        assert len(src_lines) == len(tgt_lines)
        for i in trange(len(src_lines)):
            if len(src_lines[i][:-1].split()) == 0:
                continue
            fw.write(" ".join(tokenizer.tokenize(src_lines[i][:-1])).replace(" ##", "") + " ||| " + " ".join(
                tokenizer.tokenize(tgt_lines[i][:-1])).replace(" ##", "") + " \n")


def create_parallel_dict():

    dict = {}

    with open("data/parallel_sentences.txt", "r") as fr_sent, open("data/parallel.align", "r") as fr_align:
        sents = fr_sent.readlines()
        aligns = fr_align.readlines()
        for i in trange(len(sents)):
            lg_1, lg_2 = sents[i][:-1].split("|||")
            lg_1_tokens = lg_1.split()
            lg_2_tokens = lg_2.split()

            for index in aligns[i][:-1].split():
                lg_1_index, lg_2_index = index.split("-")
                lg_1_index = int(lg_1_index)
                lg_2_index = int(lg_2_index)

                lg_1_token = lg_1_tokens[lg_1_index].lower()
                if lg_1_token not in dict:
                    dict[lg_1_token] = []
                dict[lg_1_token].append(lg_2_tokens[lg_2_index].lower())

    with open("aligned_tokens", "wb") as fw:
        pickle.dump(dict, fw)

create_parallel_dict()