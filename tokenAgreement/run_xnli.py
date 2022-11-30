from utils.utils import *
from utils.bert import BertModel, BertPooler

set_seed()
args = get_args()

LR_PRETRAIN = 0.00001
LR_FINETUNE = 0.00009
WARMUP_STEPS = 200
TRAINING_STEPS = 8000
MAX_LENGTH = 128
MODEL_NAME = "bert-base-multilingual-cased"
LANGUAGE_IDS = ["hi", "de", "ru", "tr"]

aligned_tokens = get_aligned_tokens(LANGUAGE_IDS)

print(args.ratio, args.mode)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

eval_res = {}
for _lg in LANGUAGE_IDS:
    eval_res[_lg] = []


class CrossLingualModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.xlm = BertModel.from_pretrained(MODEL_NAME)

        self.pooler = BertPooler(self.xlm.config)

        self.hidden_size = 768
        self.dropout = nn.Dropout(0.1)

        self.num_labels = 3
        self.nli_classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.loss = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None, eng_input_ids=None,
                eng_attention_mask=None, eng_token_type_ids=None, labels=None, **kwargs):

        input_ids = input_ids.to(self.xlm.device)
        attention_mask = attention_mask.to(self.xlm.device)
        token_type_ids = token_type_ids.to(self.xlm.device)
        if eng_input_ids is not None:
            eng_input_ids = eng_input_ids.to(self.xlm.device) if args.mode == "align" else None
            eng_attention_mask = eng_attention_mask.to(self.xlm.device) if args.mode == "align" else None
            eng_token_type_ids = eng_token_type_ids.to(self.xlm.device) if args.mode == "align" else None

        outputs = self.xlm(input_ids, attention_mask, token_type_ids=token_type_ids, eng_input_ids=eng_input_ids,
                           eng_attention_mask=eng_attention_mask, eng_token_type_ids=eng_token_type_ids)

        sequence_output = self.dropout(outputs)

        logits = self.pooler(sequence_output[:,:MAX_LENGTH])
        logits = self.nli_classifier(logits)

        if eng_input_ids is not None:
            logits_eng = self.pooler(sequence_output[:, MAX_LENGTH:])
            logits_eng = self.nli_classifier(logits_eng)

        loss = None
        if labels is not None:
            labels = labels.to(self.xlm.device)
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
            if eng_input_ids is not None:
                loss += self.loss(logits_eng.view(-1, self.num_labels), labels.view(-1))
                loss = loss / 2
        return loss, logits


def _process_task_data(lg):
    eval_dataset = load_dataset("xnli", lg, split="test",
                                cache_dir="/brtx/605-nvme1/yukunfeng/cross-lingual/tokenAgreement/data/huggingface")

    def preprocess_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        desc="Running tokenizer on validation dataset",
    )
    return eval_dataset


def process_task_data():
    eval_dataloaders = {}
    for lg in LANGUAGE_IDS:
        eval_dataloaders[lg] = DataLoader(_process_task_data(lg), collate_fn=default_data_collator,
                                          batch_size=64)

    return eval_dataloaders


def get_train_dataloader():
    train_dataset = load_dataset("xnli", "en", split="train", keep_in_memory=True,
                                 cache_dir="/brtx/605-nvme1/yukunfeng/cross-lingual/tokenAgreement/data/huggingface")

    train_premise = train_dataset["premise"][:]
    train_hypothesis = train_dataset["hypothesis"][:]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    new_premise = []
    new_hypothesis = []
    for i in range(len(train_premise)):
        tokenized_premise = " ".join(tokenizer.tokenize(train_premise[i])).replace(" ##", "").split()
        tokenized_hypothesis = " ".join(tokenizer.tokenize(train_hypothesis[i])).replace(" ##", "").split()

        for token_i in range(len(tokenized_premise)):
            cur_token = tokenized_premise[token_i].lower()
            if cur_token in aligned_tokens:
                if random.random() < args.ratio:
                    tokenized_premise[token_i] = random.choice(aligned_tokens[cur_token])

        for token_i in range(len(tokenized_hypothesis)):
            cur_token = tokenized_hypothesis[token_i].lower()
            if cur_token in aligned_tokens:
                if random.random() < args.ratio:
                    tokenized_hypothesis[token_i] = random.choice(aligned_tokens[cur_token])
        new_premise.append(" ".join(tokenized_premise))
        new_hypothesis.append(" ".join(tokenized_hypothesis))

    train_dataset = train_dataset.add_column("premise_new", new_premise)
    train_dataset = train_dataset.add_column("hypothesis_new", new_hypothesis)

    def preprocess_function(examples):
        tokenized_inputs = tokenizer(
            examples["premise_new"],
            examples["hypothesis_new"],
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        original_tokenized_inputs = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        tokenized_inputs["eng_input_ids"] = original_tokenized_inputs["input_ids"]
        tokenized_inputs["eng_token_type_ids"] = original_tokenized_inputs["token_type_ids"]
        tokenized_inputs["eng_attention_mask"] = original_tokenized_inputs["attention_mask"]

        return tokenized_inputs

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        keep_in_memory=True
    )

    train_dataloader = DataLoader(train_dataset, collate_fn=default_data_collator, shuffle=True,
                                  batch_size=64)

    return train_dataloader


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
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, TRAINING_STEPS)

    for epoch in range(5):
        model.train()
        all_loss = 0
        update_step = 0
        train_dataloader = get_train_dataloader()
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

    for lg in eval_res.keys():
        best_res = max(eval_res[lg])
        print(lg, best_res, eval_res[lg].index(best_res))


def evaluate(model, dataloaders):
    for lg, dataloader in dataloaders.items():
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

    eval_res[lg].append(float(acc / nums))
    print("Accuracy: {}, {}".format(acc / nums, lg))

    return acc / nums


if __name__ == "__main__":
    main()