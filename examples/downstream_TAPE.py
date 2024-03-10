import os

import torch
from transformers import BertTokenizerFast, Trainer, set_seed, HfArgumentParser, TrainingArguments

from ProteinDT.TAPE_benchmark import output_mode_mapping, dataset_processor_mapping
from ProteinDT.TAPE_benchmark import model_mapping
from ProteinDT.TAPE_benchmark import build_compute_metrics_fn
from ProteinDT.TAPE_benchmark import OntoProteinTrainer

from dataclasses import dataclass, field


@dataclass
class GeneralArguments:
    task_name: str = field(
        default="ss3",
        metadata={"help": "The name of the task to train on: " + ", ".join(dataset_processor_mapping.keys())}
    )
    data_dir: str = field(
        default="../data/downstream_datasets",
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    mean_output: bool = field(
        default=True, metadata={"help": "output of bert, use mean output or pool output"}
    )
    optimizer: str = field(
        default="AdamW",
        metadata={"help": "use optimizer: AdamW(True) or Adam(False)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    frozen_bert: bool = field(
        default=False,
        metadata={"help": "frozen bert model."}
    )
    pretrained_model: str = field(default="ProtBERT") # "ProtBERT", "ProtBERT_BFD", "OntoProtein", "ProteinDT"
    pretrained_folder: str = field(default=None)
    def __post_init__(self):
        self.task_name = self.task_name.lower()


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # For ensemble
    save_strategy: str = field(
        default='no',
        metadata={"help": "The checkpoint save strategy to adopt during training."}
    )
    evaluation_strategy: str = field(
        default='steps',
        metadata={"help": "The evaluation strategy to adopt during training."}
    )
    eval_steps: int = field(
        default=2000,
        metadata={"help": "Number of update steps between two evaluations"}
    )
    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )
    evaluate_during_training: bool = field(
        default=True,
        metadata={"help": "evaluate during training."}
    )
    fp16 = True


if __name__ == "__main__":
    parser = HfArgumentParser((GeneralArguments, DynamicTrainingArguments))
    general_args, training_args = parser.parse_args_into_dataclasses()
    print("general_args", general_args)

    set_seed(training_args.seed)

    model_name = "Rostlab/prot_bert"
    chache_dir = "../data/temp_pretrained_ProtBert"
    if general_args.pretrained_model == "ProtBERT_BFD":
        model_name = "Rostlab/prot_bert_bfd"
        chache_dir = "../data/temp_pretrained_ProtBert_BFD"
    elif general_args.pretrained_model == "OntoProtein":
        model_name = "zjukg/OntoProtein"
        chache_dir = "../data/temp_pretrained_OntoProtein"

    tokenizer = BertTokenizerFast.from_pretrained(model_name, chache_dir=chache_dir, do_lower_case=False)

    output_mode = output_mode_mapping[general_args.task_name]
    processor = dataset_processor_mapping[general_args.task_name](tokenizer)
    num_labels = len(processor.get_labels())

    train_dataset = (processor.get_train_examples(data_dir=general_args.data_dir))
    eval_dataset = (processor.get_dev_examples(data_dir=general_args.data_dir))

    if general_args.task_name == 'remote_homology':
        test_fold_dataset = (
            processor.get_test_examples(data_dir=general_args.data_dir, data_cat='test_fold_holdout')
        )
        test_family_dataset = (
            processor.get_test_examples(data_dir=general_args.data_dir, data_cat='test_family_holdout')
        )
        test_superfamily_dataset = (
            processor.get_test_examples(data_dir=general_args.data_dir, data_cat='test_superfamily_holdout')
        )

    elif general_args.task_name == 'ss3' or general_args.task_name == 'ss8':
        cb513_dataset = (
            processor.get_test_examples(data_dir=general_args.data_dir, data_cat='cb513')
        )
        ts115_dataset = (
            processor.get_test_examples(data_dir=general_args.data_dir, data_cat='ts115')
        )
        casp12_dataset = (
            processor.get_test_examples(data_dir=general_args.data_dir, data_cat='casp12')
        )
        
    else:
        test_dataset = (
            processor.get_test_examples(data_dir=general_args.data_dir, data_cat='test')
        )

    model = model_mapping[general_args.task_name].from_pretrained(
        model_name,
        cache_dir=chache_dir,
        mean_output=general_args.mean_output,
        num_labels=len(processor.get_labels()),
    )
    if general_args.pretrained_model in ["ProteinDT"]:
        assert general_args.pretrained_folder is not None
        input_model_path = os.path.join(general_args.pretrained_folder, "protein_model.pth")
        print("Loading protein model from {}...".format(input_model_path))
        state_dict = torch.load(input_model_path, map_location='cpu')
        missing_keys, unexpected_keys = model.bert.load_state_dict(state_dict, strict=False)
        print("missing keys: {}".format(missing_keys))
        print("unexpected keys: {}".format(unexpected_keys))

    if general_args.frozen_bert:
        unfreeze_layers = ['layer.29', 'bert.pooler', 'classifier']
        for name, parameters in model.named_parameters():
            parameters.requires_grad = False
            for tags in unfreeze_layers:
                if tags in name:
                    parameters.requires_grad = True
                    break

    if general_args.task_name == 'stability' or general_args.task_name == 'fluorescence':
        training_args.metric_for_best_model = "eval_spearmanr"
    elif general_args.task_name == 'remote_homology':
        training_args.metric_for_best_model = "eval_accuracy"

    if general_args.task_name == 'contact':
        # training_args.do_predict=False
        trainer = OntoProteinTrainer(
            # model_init=init_model,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(general_args.task_name, output_type=output_mode),
            data_collator=train_dataset.collate_fn,
            optimizers=(None, None)
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(general_args.task_name, output_type=output_mode),
            data_collator=train_dataset.collate_fn,
            optimizers=(None, None)
        )

    # Training
    if training_args.do_train:
        # pass
        trainer.train()
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    # trainer.compute_metrics = metrics_mapping(args.task_name)
    if general_args.task_name == 'remote_homology':
        predictions_fold_family, input_ids_fold_family, metrics_fold_family = trainer.predict(test_fold_dataset)
        predictions_family_family, input_ids_family_family, metrics_family_family = trainer.predict(test_family_dataset)
        predictions_superfamily_family, input_ids_superfamily_family, metrics_superfamily_family = trainer.predict(test_superfamily_dataset)
        print("metrics_fold: ", metrics_fold_family)
        print("metrics_family: ", metrics_family_family)
        print("metrics_superfamily: ", metrics_superfamily_family)
    elif general_args.task_name == 'ss8' or general_args.task_name == 'ss3':
        predictions_cb513, input_ids_cb513, metrics_cb513 = trainer.predict(cb513_dataset)
        predictions_ts115, input_ids_ts115, metrics_ts115 = trainer.predict(ts115_dataset)
        predictions_casp12, input_ids_casp12, metrics_casp12 = trainer.predict(casp12_dataset)
        print("cb513: ", metrics_cb513)
        print("ts115: ", metrics_ts115)
        print("casp12: ", metrics_casp12)
    else:
        predictions_family, input_ids_family, metrics_family = trainer.predict(test_dataset)
        print("metrics", metrics_family)