import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import string

import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer, AutoTokenizer, OPTForCausalLM
from torch.utils.data import DataLoader

from ProteinDT.datasets import MISATODataset

from utils import text_prompt_dict
from utils_peptide_editing import ProteinBindingDataset, load_oracle_evaluator, evaluate_folding, evaluate_binding, save_PDB_list

import re


def parse_Galatica_result(text_prompt, result):
    result = result.replace(text_prompt, "")
    result = result.split("[END_AMINO]")[0].strip()
    result = result.split("\n")[0].replace("?", "")

    pattern = re.compile('[A-Z]{5,}')
    parsed_results = pattern.findall(result)

    result = parsed_results[0]
    result = re.sub(r"[UZOB]", "X", result)

    return result


@torch.no_grad()
def inference_Galactica(dataloader):
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader

    galactica_tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-1.3b")
    galactica_model = OPTForCausalLM.from_pretrained("facebook/galactica-1.3b", device_map="auto")

    input_protein_sequence_list, edited_protein_sequence_list, PDB_id_list = [], [], []
    for batch_idx, batch in enumerate(L):
        peptide_sequence_batch, text_sequence_batch, PDB_id_batch = batch["peptide_sequence"], batch["text_sequence"], batch["PDB_id"]
        for peptide_sequence, text_sequence, PDB_id in zip(peptide_sequence_batch, text_sequence_batch, PDB_id_batch):
            peptide_sequence = ''.join(peptide_sequence).replace(" ", "")
            text_sequence = ''.join(text_sequence).replace(" ", "")

            text_prompt = "Given an input peptide amino acid sequence [START_AMINO]{}[END_AMINO] and a target protein. The target protein satisfies the following property. {} Can you {}, which needs to be similar to the input sequence? [START_AMINO]".format(
                peptide_sequence, text_sequence, text_prompt_dict[args.editing_task][args.text_prompt_id])
            text_ids = galactica_tokenizer(text_prompt, return_tensors="pt").input_ids.to("cuda")
            print("text_prompt", text_prompt)

            outputs = galactica_model.generate(
                text_ids,
                max_new_tokens=len(peptide_sequence)+10,
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
            try:
                edited_protein_sequence = parse_Galatica_result(text_prompt, protein_sequence_answer)
            except:
                print("invalid", protein_sequence_answer)
                continue

            input_protein_sequence_list.append(peptide_sequence)
            edited_protein_sequence_list.append(edited_protein_sequence)
            PDB_id_list.append(PDB_id)

    return input_protein_sequence_list, edited_protein_sequence_list, PDB_id_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mutation_number", type=int, default=1)

    parser.add_argument("--editing_task", type=str, default="peptide_binding")    
    parser.add_argument("--dataset_size", type=int, default=200)
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
    dataset = ProteinBindingDataset(
        data_folder=text_prompt_dict[args.editing_task]["data_folder"],
        dataset_size=args.dataset_size,
        protein_tokenizer=CLAP_protein_tokenizer,
        protein_max_sequence_len=args.protein_max_sequence_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    input_peptide_sequence_list, edited_peptide_sequence_list, PDB_id_list = inference_Galactica(dataloader)

    if args.output_folder is None:
        exit()
    
    ##### Load pretrained model
    eval_prediction_model = load_oracle_evaluator(device)
    data_folder = "../../data/MISATO"
    misato_dataset = MISATODataset(data_folder)
    peptide_idx2data = misato_dataset.get_peptide_idx2data()

    print('input:')
    input_peptide_PDB_id_list = evaluate_folding(input_peptide_sequence_list, eval_prediction_model)
    PDB_output_folder = os.path.join(args.output_folder, "input_PDB")
    os.makedirs(PDB_output_folder, exist_ok = True)
    save_PDB_list(input_peptide_PDB_id_list, PDB_id_list, PDB_output_folder)
    input_eval_list = evaluate_binding(input_peptide_PDB_id_list, PDB_id_list, peptide_idx2data, eval_prediction_model, device, args)

    print("output")
    output_peptide_PDB_id_list = evaluate_folding(edited_peptide_sequence_list, eval_prediction_model)
    PDB_output_folder = os.path.join(args.output_folder, "output_PDB")
    os.makedirs(PDB_output_folder, exist_ok = True)
    save_PDB_list(output_peptide_PDB_id_list, PDB_id_list, PDB_output_folder)
    output_eval_list = evaluate_binding(output_peptide_PDB_id_list, PDB_id_list, peptide_idx2data, eval_prediction_model, device, args)

    if args.output_text_file_path is None:
        args.output_text_file_path = os.path.join(args.output_folder, "step_01_editing.txt")

    f = open(args.output_text_file_path, 'w')
    hit, total = 0, 0
    for input_peptide, input_eval, edited_peptide, output_eval in zip(input_peptide_sequence_list, input_eval_list, edited_peptide_sequence_list, output_eval_list):
        print('input,{},{}'.format(input_peptide, input_eval), file=f)
        print('output,{},{}'.format(edited_peptide, output_eval), file=f)

        total += 1
        
        if args.editing_task in ["peptide_binding"]:
            if args.text_prompt_id in [101]:
                if output_eval < input_eval:
                    hit += 1
            elif args.text_prompt_id in [201]:
                if output_eval > input_eval:
                    hit += 1

    hit_ratio = 100. * hit / total
    print("hit: {}".format(hit))
    print("#1 total: {}".format(total))
    print("#1 hit ratio: {}".format(hit_ratio))
    
    total = len(dataset)
    hit_ratio = 100. * hit / total
    print("total: {}".format(total))
    print("hit ratio: {}".format(hit_ratio))