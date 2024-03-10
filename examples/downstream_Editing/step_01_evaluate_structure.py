import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import string
import mdtraj
import re

import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader

from utils import ProteinDataset, ProteinSeqDataset, text_prompt_dict, load_oracle_evaluator, evaluate, analyze

from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37


@torch.no_grad()
def save_and_evaluate_PDB_list(PDB_list, idx_list, PDB_output_folder):
    DSSP_list = []
    valid_idx_set = set()
    for PDB, idx in zip(PDB_list, idx_list):
        file_name = os.path.join(PDB_output_folder, "{}.pdb".format(idx))
        f = open(file_name, "w")
        f.write("".join(PDB))
        f.flush()
        f.close()
        
        try:
            # MDtraj
            traj = mdtraj.load(file_name)
            # SS calculation
            pdb_ss = mdtraj.compute_dssp(traj, simplified=True)[0]  # (L, )

            ss_count = {
                'alpha': np.sum(pdb_ss == 'H'),
                'beta': np.sum(pdb_ss == 'E'),
            }
            print("ss_count", ss_count)
            DSSP_list.append(ss_count)
            valid_idx_set.add(idx)
        except Exception as e:
            print(f'[Warning] Mdtraj failed with error {e}')
    return DSSP_list, valid_idx_set


def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdb = to_pdb(pred)
        pdbs.append(pdb)
    return pdbs


@torch.no_grad()
def evaluate_folding(protein_sequence_list):
    PDB_data_list, idx_list = [], []
    for idx, protein_sequence in enumerate(tqdm(protein_sequence_list)):

        tokenized_input = folding_tokenizer(protein_sequence, return_tensors="pt", add_special_tokens=False)['input_ids']
        tokenized_input = tokenized_input.cuda()

        output = folding_model(tokenized_input)

        PDB_list = convert_outputs_to_pdb(output)

        PDB_data_list.extend(PDB_list)
        idx_list.append(idx)

    return PDB_data_list, idx_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mutation_number", type=int, default=1)

    parser.add_argument("--editing_task", type=str, default="alpha")    
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

    assert args.editing_task in ["alpha", "beta"]

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    folding_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    folding_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to(device)
    folding_model.trunk.set_chunk_size(512)
    
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

    if args.output_text_file_path is None:
        args.output_text_file_path = os.path.join(args.output_folder, "step_01_editing.txt")
    
    f = open(args.output_text_file_path, "r")
    input_protein_sequence_list, input_eval_list, edited_protein_sequence_list, edited_eval_list = [], [], [], []
    for line in f.readlines():
        if line.startswith("input"):
            line = line.strip().split(",")
            input_protein_sequence_list.append(line[1])
            value = line[2].replace("[", "").replace("]", "")
            input_eval_list.append(float(value))
        elif line.startswith("output"):
            line = line.strip().split(",")
            edited_protein_sequence = line[1]
            edited_protein_sequence = re.sub(r"[UZOB]", "X", edited_protein_sequence)
            edited_protein_sequence_list.append(edited_protein_sequence)
            value = line[2].replace("[", "").replace("]", "")
            edited_eval_list.append(float(value))

    neo_input_protein_sequence_list, neo_input_eval_list, neo_edited_protein_sequence_list, neo_edited_eval_list = [], [], [], []
    for a,b,c,d in zip(input_protein_sequence_list, input_eval_list, edited_protein_sequence_list, edited_eval_list):
        if len(c) == 0:
            continue
        if len(a) >= 512:
            continue
        if len(c) >= 512:
            continue
        neo_input_protein_sequence_list.append(a)
        neo_input_eval_list.append(b)
        neo_edited_protein_sequence_list.append(c)
        neo_edited_eval_list.append(d)
    input_protein_sequence_list, input_eval_list, edited_protein_sequence_list, edited_eval_list = neo_input_protein_sequence_list, neo_input_eval_list, neo_edited_protein_sequence_list, neo_edited_eval_list

    #################### compare ####################
    eval_hit, total = 0, 0
    for input_protein, input_eval, edited_protein, output_eval in zip(input_protein_sequence_list, input_eval_list, edited_protein_sequence_list, edited_eval_list):
        total += 1
        
        if args.text_prompt_id in [101]:
            if output_eval > input_eval:
                eval_hit += 1
        elif args.text_prompt_id in [201]:
            if output_eval < input_eval:
                eval_hit += 1
    if total > 0:
        eval_hit_ratio = 100. * eval_hit / total
        print("#1 eval hit: {}".format(eval_hit))
        print("#1 eval total: {}".format(total))
        print("#1 eval hit ratio: {}".format(eval_hit_ratio))

    total = len(dataset)
    eval_hit_ratio = 100. * eval_hit / total
    print("eval hit: {}".format(eval_hit))
    print("eval total: {}".format(total))
    print("eval hit ratio: {}".format(eval_hit_ratio))
    ##################################################

    input_PDB_list, idx_list = evaluate_folding(input_protein_sequence_list)
    PDB_output_folder = os.path.join(args.output_folder, "input_PDB")
    os.makedirs(PDB_output_folder, exist_ok = True)
    input_DSSP_list, input_valid_idx_set = save_and_evaluate_PDB_list(input_PDB_list, idx_list, PDB_output_folder)
    print("valid input (w/ secondary structure)", len(input_valid_idx_set))

    output_PDB_list, idx_list = evaluate_folding(edited_protein_sequence_list)
    PDB_output_folder = os.path.join(args.output_folder, "output_PDB")
    os.makedirs(PDB_output_folder, exist_ok = True)
    output_DSSP_list, output_valid_idx_set = save_and_evaluate_PDB_list(output_PDB_list, idx_list, PDB_output_folder)
    print("valid output (w/ secondary structure)", len(output_valid_idx_set))

    ##### compare
    evaluation_output_file_path = os.path.join(args.output_folder, "step_01_evaluate.txt")
    f = open(evaluation_output_file_path, 'w')
    eval_hit, DSSP_hit, total = 0, 0, 0
    for input_protein, input_eval, input_DSSP, edited_protein, output_eval, output_DSSP, idx in zip(input_protein_sequence_list, input_eval_list, input_DSSP_list, edited_protein_sequence_list, edited_eval_list, output_DSSP_list, idx_list):
        if not (idx in input_valid_idx_set and idx in output_valid_idx_set):
            print("invalid", idx, input_valid_idx_set, output_valid_idx_set)
            continue
        print('input,{},{},{}'.format(input_protein, input_eval, input_DSSP), file=f)
        print('output,{},{},{}'.format(edited_protein, output_eval, output_DSSP), file=f)

        total += 1
        
        if args.text_prompt_id in [101]:
            if output_DSSP[args.editing_task] > input_DSSP[args.editing_task]:
                DSSP_hit += 1
        elif args.text_prompt_id in [201]:
            if output_DSSP[args.editing_task] < input_DSSP[args.editing_task]:
                DSSP_hit += 1

    if total > 0:
        DSSP_hit_ratio = 100. * DSSP_hit / total
        print("#1 DSSP hit: {}".format(DSSP_hit))
        print("#1 DSSP total: {}".format(total))
        print("#1 DSSP hit ratio: {}".format(DSSP_hit_ratio))

    total = len(dataset)
    DSSP_hit_ratio = 100. * DSSP_hit / total
    print("DSSP hit: {}".format(DSSP_hit))
    print("DSSP total: {}".format(total))
    print("DSSP hit ratio: {}".format(DSSP_hit_ratio))

    """
    ../../output/ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-EBM_NCE-0.1-batch-9-gpu-8-epoch-5/step_06_AE_1e-3_3/downstream_Editing_latent_optimization/alpha_prompt_101_lambda_0.1_num_repeat_16_oracle_text_T_2
    cp ../../output/ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-EBM_NCE-0.1-batch-9-gpu-8-epoch-5/step_06_AE_1e-3_3/downstream_Editing_latent_optimization/alpha_prompt_101_lambda_0.1_num_repeat_16_oracle_text_T_2 .

    python step_01_evaluate_structure.py --num_workers=0 --batch_size=4 --editing_task=alpha --text_prompt_id=101  --output_folder=alpha_prompt_101_lambda_0.1_num_repeat_16_oracle_text_T_2 --output_text_file_path=alpha_prompt_101_lambda_0.1_num_repeat_16_oracle_text_T_2/step_01_editing.txt
    """
