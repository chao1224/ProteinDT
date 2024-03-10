import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import string
import re

import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer, AutoTokenizer, OPTForCausalLM
from torch.utils.data import DataLoader

from utils import ProteinDataset, ProteinSeqDataset, text_prompt_dict, load_oracle_evaluator, evaluate, analyze


def parse_Galatica_result(text_prompt, result):
    result = result.replace(text_prompt, "")
    result = result.split("[END_AMINO]")[0].strip()
    result = result.split("\n")[0].replace("?", "")

    pattern = re.compile('[A-Z]{5,}')
    parsed_results = pattern.findall(result)
    if len(parsed_results) == 0:
        print("invalid")
        return ""

    result = parsed_results[0]
    result = re.sub(r"[UZOB]", "X", result)
    return result


@torch.no_grad()
def inference_Galactica(dataloader, mutation_number):
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader

    galactica_tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-1.3b")
    galactica_model = OPTForCausalLM.from_pretrained("facebook/galactica-1.3b", device_map="auto")

    input_protein_sequence_list, edited_protein_sequence_list = [], []
    for batch_idx, batch in enumerate(L):
        protein_sequence_batch = batch["protein_sequence"]
        for protein_sequence in protein_sequence_batch:
            # protein_sequence = protein_sequence.split(" ")
            protein_sequence = ''.join(protein_sequence).replace(" ", "")

            text_prompt = "Given an input amino acid sequence [START_AMINO]{}[END_AMINO]. Can you {}, which needs to be similar to the input sequence? [START_AMINO]".format(
                protein_sequence, text_prompt_dict[args.editing_task][args.text_prompt_id])
            text_ids = galactica_tokenizer(text_prompt, return_tensors="pt").input_ids.to("cuda")

            outputs = galactica_model.generate(
                text_ids,
                max_new_tokens=len(protein_sequence)+10,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                use_cache=True,
                top_k=5,
                repetition_penalty=1.0,
                length_penalty=1,
                num_return_sequences=1,
            )

            output = outputs[0]
            protein_sequence_answer = galactica_tokenizer.decode(output)
            edited_protein_sequence = parse_Galatica_result(text_prompt, protein_sequence_answer)
            if len(edited_protein_sequence) == 0:
                continue

            input_protein_sequence_list.append(protein_sequence)
            edited_protein_sequence_list.append(edited_protein_sequence)

    return input_protein_sequence_list, edited_protein_sequence_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mutation_number", type=int, default=1)

    parser.add_argument("--editing_task", type=str, default="Villin")    
    parser.add_argument("--dataset_size", type=int, default=None)
    parser.add_argument("--text_prompt_id", type=int, default=101)

    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--output_text_file_path", type=str, default=None)

    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)

    parser.add_argument("--protein_backbone_model", type=str, default="ProtBERT_BFD", choices=["ProtBERT", "ProtBERT_BFD"])
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)

    args = parser.parse_args()
    print("arguments", args)

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    ##### Load pretrained protein model
    if args.protein_backbone_model == "ProtBERT":
        CLAP_protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    elif args.protein_backbone_model == "ProtBERT_BFD":
        CLAP_protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
    protein_dim = 1024
    
    ##### load protein sequence
    dataset_file_path = os.path.join(text_prompt_dict[args.editing_task]["data_folder"], "preprocessed_data.csv")
    dataset = ProteinDataset(
        dataset_file_path=dataset_file_path,
        dataset_size=args.dataset_size,
        protein_tokenizer=CLAP_protein_tokenizer,
        protein_max_sequence_len=args.protein_max_sequence_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    input_protein_sequence_list, edited_protein_sequence_list = inference_Galactica(dataloader, mutation_number=args.mutation_number)

    if args.output_folder is None:
        exit()
    
    ##### Load pretrained model
    eval_prediction_model, eval_protein_tokenizer = load_oracle_evaluator(args.editing_task, device)

    print('input:')
    input_dataset = ProteinSeqDataset(input_protein_sequence_list, eval_protein_tokenizer, args.protein_max_sequence_len)
    input_dataloader = DataLoader(input_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    input_eval_list = evaluate(input_dataloader, eval_prediction_model, device, args)
    file_path = os.path.join(args.output_folder, "{editing_task}_input.png".format(editing_task=args.editing_task))
    analyze(input_eval_list, args, file_path)

    print("output")
    output_dataset = ProteinSeqDataset(edited_protein_sequence_list, eval_protein_tokenizer, args.protein_max_sequence_len)
    output_dataloader = DataLoader(output_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    output_eval_list = evaluate(output_dataloader, eval_prediction_model, device, args)
    file_path = os.path.join(args.output_folder, "{editing_task}_output_{mutation_number}.png".format(editing_task=args.editing_task, mutation_number=args.mutation_number))
    analyze(output_eval_list, args, file_path)


    if args.output_text_file_path is None:
        args.output_text_file_path = os.path.join(args.output_folder, "step_01_editing.txt")

    f = open(args.output_text_file_path, 'w')
    hit, total = 0, 0
    for input_protein, input_eval, edited_protein, output_eval in zip(input_protein_sequence_list, input_eval_list, edited_protein_sequence_list, output_eval_list):
        print('input,{},{}'.format(input_protein, input_eval), file=f)
        print('output,{},{}'.format(edited_protein, output_eval), file=f)

        total += 1
        
        if args.editing_task in ["alpha", "beta"]:
            if args.text_prompt_id in [101]:
                if output_eval > input_eval:
                    hit += 1
            elif args.text_prompt_id in [201]:
                if output_eval < input_eval:
                    hit += 1

        elif args.editing_task in ["Villin", "Pin1", "hYAP65"]:
            if args.text_prompt_id in [101, 102]:
                if output_eval < input_eval:
                    hit += 1
            elif args.text_prompt_id in [201, 202]:
                if output_eval > input_eval:
                    hit += 1

    hit_ratio = 100. * hit / total
    print("hit: {}".format(hit))
    print("total: {}".format(total))
    print("hit ratio: {}".format(hit_ratio))
