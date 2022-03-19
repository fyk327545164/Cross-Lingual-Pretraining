from tqdm import trange
from transformers import AutoTokenizer
import pickle
import os



class Aligner:
    def __init__(self, src_lang, tgt_lang, data_root=None, tokenizer_model="bert-base-multilingual-cased"):
        self.data_root = "../data/" if data_root is None else data_root
        self.parallel_data_file_name = None
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.lang_pair = src_lang + '-' + tgt_lang if os.path.isdir(
            self.data_root + src_lang + '-' + tgt_lang) else tgt_lang + '-' + src_lang
        self.tokenizer_name = tokenizer_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.parallel_data_folder = f"{self.data_root}{self.lang_pair}/"


    def create_parallel_data(self):
        filename_src = f"{self.parallel_data_folder}News-Commentary.{self.lang_pair}.{self.src_lang}"
        filename_tgt = f"{self.parallel_data_folder}News-Commentary.{self.lang_pair}.{self.tgt_lang}"
        filename_res = f"{self.parallel_data_folder}{self.lang_pair}_parallel-sentences_{self.tokenizer_name}.txt"

        if os.path.exists(filename_res):
            with open(filename_src, 'r') as src_fr, open(filename_res, "r") as res_fr:
                src_lines = src_fr.readlines()
                res_lines = res_fr.readlines()
                if len(src_lines) == len(res_lines):
                    self.parallel_data_file_name = filename_res
                    return

        with open(filename_src, 'r') as fr_src, \
                open(filename_tgt, 'r') as fr_tgt, \
                open(filename_res, "w") as fw:

            src_lines = fr_src.readlines()
            tgt_lines = fr_tgt.readlines()
            assert len(src_lines) == len(tgt_lines)
            for i in trange(len(src_lines)):
                if len(src_lines[i][:-1].split()) == 0:
                    continue
                src_tokens = self.tokenizer.tokenize(src_lines[i][:-1])
                tgt_tokens = self.tokenizer.tokenize(tgt_lines[i][:-1])

                src_string = " ".join(src_tokens).replace(" ##", "")
                tgt_string = " ".join(tgt_tokens).replace(" ##", "")

                fw.write(f"{src_string} ||| {tgt_string} \n")
        self.parallel_data_file_name = filename_res

    def run_fast_align(self):
        cur_root = os.getcwd()
        res = f""
        os.system(f"cd {self.data_root}")
        os.system("git clone https://github.com/clab/fast_align")
        os.system('cd fast_align')
        os.system('mkdir build')
        os.system('cd build')
        os.system('cmake ..')
        os.system('make')
        # os.system("cd ../../")
        os.system(f'./fast_align -i {self.parallel_data_file_name} -d -o -v > {res}')

    def create_parallel_match_table(self, fast_align_res=None):
        if fast_align_res is None:
            fast_align_res = f"{self.parallel_data_folder}{self.lang_pair}_{self.tokenizer_name}.align"
        dict = {}
        align_table_file_path = f"{self.parallel_data_folder}{self.lang_pair}_{self.tokenizer_name}_table"

        with open(self.parallel_data_file_name, "r") as fr_sent, open(fast_align_res, "r") as fr_align:
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
                    #f lg_1_token not in dict:
                    #    dict[lg_1_token] = []
                    lg_2_token = lg_2_tokens[lg_2_index]
                    if lg_1_token != lg_2_token:
                        if lg_1_token not in dict:
                            dict[lg_1_token] = []
                        dict[lg_1_token].append(lg_2_token)

    with open("aligned_tokens", "wb") as fw:
        pickle.dump(dict, fw)



en_ru_aligner = Aligner("en", "ru")
en_ru_aligner.create_parallel_data()
en_ru_aligner.create_parallel_match_table()
