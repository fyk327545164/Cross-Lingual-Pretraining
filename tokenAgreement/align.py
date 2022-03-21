from datasets import load_dataset
from tqdm import trange, tqdm
from transformers import AutoTokenizer
import pickle
import os

lg_corpus = {"de": "wmt14", "hi": "wmt14", "ru": "wmt14", "zh": "wmt19", "tr": "wmt16"}


class Aligner:
    def __init__(self, src_lang, tgt_lang, data_root=None, tokenizer_model="bert-base-multilingual-cased"):
        self.data_root = "../data/" if data_root is None else data_root
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.parallel_data_file_name = "data/parallel-{}-{}.txt".format(self.src_lang, self.tgt_lang)
        self.parallel_index_file_name = "data/aligned-{}-{}-index.txt".format(self.src_lang, self.tgt_lang)
        self.aligned_tokens_file_name = "data/aligned-tokens-{}-{}".format(self.src_lang, self.tgt_lang)

        self.lang_pair = src_lang + '-' + tgt_lang if os.path.isdir(
            self.data_root + src_lang + '-' + tgt_lang) else tgt_lang + '-' + src_lang
        self.tokenizer_name = tokenizer_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.parallel_data_folder = f"{self.data_root}{self.lang_pair}/"

    def create_parallel_data(self):

        ds = load_dataset(lg_corpus[self.tgt_lang], "{}-en".format(self.tgt_lang),
                          cache_dir="/brtx/605-nvme1/yukunfeng/cross-lingual/tokenAgreement/data/huggingface")
        with open(self.parallel_data_file_name, "w") as fw:
            for lg_pair in tqdm(ds["train"]["translation"]):
                fw.write(" ".join(self.tokenizer.tokenize(lg_pair["en"])) + " ||| " + " ".join(
                    self.tokenizer.tokenize(lg_pair[self.tgt_lang])) + " \n")

    # def create_parallel_data(self):
    #     filename_src = f"{self.parallel_data_folder}News-Commentary.{self.lang_pair}.{self.src_lang}"
    #     filename_tgt = f"{self.parallel_data_folder}News-Commentary.{self.lang_pair}.{self.tgt_lang}"
    #     filename_res = f"{self.parallel_data_folder}{self.lang_pair}_parallel-sentences_{self.tokenizer_name}.txt"
    #
    #     if os.path.exists(filename_res):
    #         with open(filename_src, 'r') as src_fr, open(filename_res, "r") as res_fr:
    #             src_lines = src_fr.readlines()
    #             res_lines = res_fr.readlines()
    #             if len(src_lines) == len(res_lines):
    #                 self.parallel_data_file_name = filename_res
    #                 return
    #
    #     with open(filename_src, 'r') as fr_src, \
    #             open(filename_tgt, 'r') as fr_tgt, \
    #             open(filename_res, "w") as fw:
    #
    #         src_lines = fr_src.readlines()
    #         tgt_lines = fr_tgt.readlines()
    #         assert len(src_lines) == len(tgt_lines)
    #         for i in trange(len(src_lines)):
    #             if len(src_lines[i][:-1].split()) == 0:
    #                 continue
    #             src_tokens = self.tokenizer.tokenize(src_lines[i][:-1])
    #             tgt_tokens = self.tokenizer.tokenize(tgt_lines[i][:-1])
    #
    #             src_string = " ".join(src_tokens).replace(" ##", "")
    #             tgt_string = " ".join(tgt_tokens).replace(" ##", "")
    #
    #             fw.write(f"{src_string} ||| {tgt_string} \n")
    #     self.parallel_data_file_name = filename_res

    def run_fast_align(self):
        # cur_root = os.getcwd()
        # res = f""
        # os.system(f"cd {self.data_root}")
        # os.system("git clone https://github.com/clab/fast_align")
        # os.system('cd fast_align')
        # os.system('mkdir build')
        # os.system('cd build')
        # os.system('cmake ..')
        # os.system('make')
        # # os.system("cd ../../")
        os.system(f' ./fast_align/build/fast_align -i {self.parallel_data_file_name} -d -o -v -I 8 > {self.parallel_index_file_name}')

    def create_parallel_match_table(self):
        aligned_dict = {}
        with open(self.parallel_data_file_name, "r") as fr_sent, open(self.parallel_index_file_name, "r") as fr_align:
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

                    lg_2_token = lg_2_tokens[lg_2_index]
                    if lg_1_token != lg_2_token:
                        if lg_1_token not in aligned_dict:
                            aligned_dict[lg_1_token] = []
                        aligned_dict[lg_1_token].append(lg_2_token)

        with open(self.aligned_tokens_file_name, "wb") as fw:
            pickle.dump(aligned_dict, fw)


for lg in ["ru", "hi", "zh", "tr", "de"]:
    aligner = Aligner("en", lg)
    aligner.create_parallel_data()
    aligner.run_fast_align()
    aligner.create_parallel_match_table()
