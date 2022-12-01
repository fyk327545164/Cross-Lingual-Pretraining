import argparse
import logging
import math
import os
import pickle
import random
from pathlib import Path

import datasets
import numpy as np
import torch
from datasets import load_dataset, load_metric,concatenate_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils_qa import postprocess_qa_predictions

# all_lang_list  = ["ar","de","el","en","es","hi","ro","ru","th","tr","vi","zh"]
all_lang_list = ["ar", "de", "el", "en", "es", "hi", "ro", "ru", "tr", "vi", "zh"]

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.18.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")
logger = logging.getLogger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

check_point_path = ""


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    parser.add_argument('--ratio', required=True, type=int)
    parser.add_argument("--replace_table_file", type=str, default=None)
    parser.add_argument('--loss_interval', required=True, type=int)
    parser.add_argument('--eval_interval', default=-1, type=int)
    parser.add_argument("--output_log_file", type=str, required=True)
    parser.add_argument("--eval_lang", type=str, required=True)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=4, help="A csv or a json file containing the training data."
    )
    parser.add_argument("--do_predict", action="store_true", help="To do prediction on the question answering model")
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the Prediction data."
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=384,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
             " sequences shorter will be padded if `--pad_to_max_lengh` is passed.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate_pre_train",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_fine_tune",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="The threshold used to select the null answer: if the best answer has a score that is less than "
             "the score of the null answer minus this threshold, the null answer is selected for this example. "
             "Only useful when `version_2_with_negative=True`.",
    )
    parser.add_argument(
        "--version_2_with_negative",
        type=bool,
        default=False,
        help="If true, some of the examples do not have an answer.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help="The maximum length of an answer that can be generated. This is needed because the start "
             "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this "
             "value if set.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of evaluation examples to this "
             "value if set.",
    )
    parser.add_argument(
        "--max_predict_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of prediction examples to this",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    args = parser.parse_args()
    if (
            args.dataset_name is None
            and args.train_file is None
            and args.validation_file is None
            and args.test_file is None
    ):
        raise ValueError("Need either a dataset name or a training/validation/test file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
    return args


def main():
    args = parse_args()





    accelerator = Accelerator()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

        # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Downloading and loading a dataset from the hub.
    raw_datasets = None

    datasets_all = {}
    for lang in all_lang_list:
        datasets_all[lang] = load_dataset("xquad", f"xquad.{lang}")["validation"]

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if args.model_name_or_path:
        model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForQuestionAnswering.from_config(config)

    # Preprocessing the datasets.
    # Preprocessing is slight different for training and evaluation.
    column_names = raw_datasets["train"].column_names
    valid_column_names = datasets_all["ar"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    model.load_state_dict(torch.load(check_point_path))

    for lang in datasets_all:

        eval_examples = datasets_all[lang]
        if args.max_eval_samples != 0 :
            # We will select sample from whole data
            eval_examples = eval_examples.select(range(args.max_eval_samples))
        # Validation Feature Creation

        with accelerator.main_process_first():

            eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=valid_column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

        if args.max_eval_samples != 0 :
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))



        # DataLoaders creation:
        if args.pad_to_max_length:
            data_collator = default_data_collator
        else:
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

        eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
        eval_dataloader = DataLoader(
            eval_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )


        # Post-processing:
        def post_processing_function(examples, features, predictions, stage="eval"):
            # Post-processing: we match the start logits and end logits to answers in the original context.
            predictions = postprocess_qa_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                version_2_with_negative=args.version_2_with_negative,
                n_best_size=args.n_best_size,
                max_answer_length=args.max_answer_length,
                null_score_diff_threshold=args.null_score_diff_threshold,
                output_dir=args.output_dir,
                prefix=stage,
            )
            # Format the result to the format the metric expects.
            if args.version_2_with_negative:
                formatted_predictions = [
                    {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
                ]
            else:
                formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

            references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

        metric = load_metric("squad_v2" if args.version_2_with_negative else "squad")

        # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
        def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
            """
            Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

            Args:
                start_or_end_logits(:obj:`tensor`):
                    This is the output predictions of the model. We can only enter either start or end logits.
                eval_dataset: Evaluation dataset
                max_len(:obj:`int`):
                    The maximum length of the output tensor. ( See the model.eval() part for more details )
            """

            step = 0
            # create a numpy array and fill it with -100.
            logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
            # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather
            for i, output_logit in enumerate(start_or_end_logits):  # populate columns
                # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
                # And after every iteration we have to change the step

                batch_size = output_logit.shape[0]
                cols = output_logit.shape[1]

                if step + batch_size < len(dataset):
                    logits_concat[step: step + batch_size, :cols] = output_logit
                else:
                    logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

                step += batch_size

            return logits_concat

        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

        def run_eval():
            # Evaluation
            logger.info("\n***** Running Evaluation *****")
            logger.info(f"  Num examples = {len(eval_dataset)}")
            logger.info(f"  Batch size = {args.per_device_eval_batch_size}")
            model.eval()
            all_start_logits = []
            all_end_logits = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                    start_logits = outputs.start_logits
                    end_logits = outputs.end_logits

                    if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                        start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                        end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

                    all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
                    all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

            max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

            # concatenate the numpy array
            start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
            end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

            # delete the list of numpy arrays
            del all_start_logits
            del all_end_logits

            outputs_numpy = (start_logits_concat, end_logits_concat)
            prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)
            eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
            logger.info(f"Evaluation metrics: {eval_metric}")
            with open(args.output_log_file, "a") as log_file_fr:
                log_file_fr.write(f"eval:-------")
                log_file_fr.write(f"\n Evaluation metrics: {eval_metric}")
                log_file_fr.write(f"train:-----")

            print(lang,eval_metric)


        run_eval()









    if __name__ == "__main__":
        main()
