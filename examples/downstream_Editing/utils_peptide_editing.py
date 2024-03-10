from tqdm import tqdm
import numpy as np
import os
import json
from Bio.PDB.Polypeptide import three_to_one

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from ProteinDT.models import FoldingBindingInferenceModel


class ProteinBindingDataset(Dataset):
    def __init__(
        self, data_folder,
        protein_tokenizer, protein_max_sequence_len,
        text_tokenizer=None, text_max_sequence_len=None, text_prompt=None,
        dataset_size=None
    ):
        self.dataset_file_path = os.path.join(data_folder, "preprocessed_data.csv")
        self.PDB_mapping_file_path = os.path.join(data_folder, "PDB_mapping_data.txt")
        self.PDB_idx2text_file_path = os.path.join(data_folder, "PDB_idx2text.json")
        f = open(self.PDB_idx2text_file_path, "r")
        PDB_idx2text = json.load(f)
        self.dataset_size = dataset_size

        self.PDB_id2peptide_seq, self.peptide_seq2PDB_id, self.PDB_id2protein_seq = {}, {}, {}
        f = open(self.PDB_mapping_file_path, 'r')
        for line in f.readlines():
            line = line.strip().split(',')
            PDB_id, peptide_seq, protein_seq = line[0], line[1], line[2]
            self.PDB_id2peptide_seq[PDB_id] = peptide_seq
            self.peptide_seq2PDB_id[peptide_seq] = PDB_id
            self.PDB_id2protein_seq[PDB_id] = protein_seq

        f = open(self.dataset_file_path, 'r')
        peptide_sequence_list, protein_sequence_list, PDB_id_list, text_sequence_list = [], [], [], []
        for line in f.readlines():
            line = line.strip().split(',')
            peptide_sequence = line[0]
            peptide_sequence = peptide_sequence.replace(" ", "")
            PDB_id = self.peptide_seq2PDB_id[peptide_sequence]
            peptide_sequence = " ".join(peptide_sequence)
            peptide_sequence_list.append(peptide_sequence)
            PDB_id_list.append(PDB_id)
            protein_sequence_list.append(self.PDB_id2protein_seq[PDB_id])
            text = PDB_idx2text[PDB_id]
            text = text.split(".")[0]+"."
            if text_prompt is not None:
                text = text_prompt.format(text)
            text_sequence_list.append(text)
        self.peptide_sequence_list = peptide_sequence_list
        self.protein_sequence_list = protein_sequence_list
        self.PDB_id_list = PDB_id_list
        self.text_sequence_list = text_sequence_list

        if self.dataset_size is not None:
            self.peptide_sequence_list = self.peptide_sequence_list[:self.dataset_size]
            self.PDB_id_list = self.PDB_id_list[:self.dataset_size]

        self.protein_tokenizer = protein_tokenizer
        self.protein_max_sequence_len = protein_max_sequence_len
        self.text_tokenizer = text_tokenizer
        self.text_max_sequence_len = text_max_sequence_len
        
        return

    def __getitem__(self, index):
        peptide_sequence = self.peptide_sequence_list[index]
        protein_sequence = self.protein_sequence_list[index]
        text_sequence = self.text_sequence_list[index]
        PDB_id = self.PDB_id_list[index]
        
        peptide_sequence_encode = self.protein_tokenizer(peptide_sequence, truncation=True, max_length=self.protein_max_sequence_len, padding='max_length', return_tensors='pt')
        peptide_sequence_input_ids = peptide_sequence_encode.input_ids.squeeze()
        peptide_sequence_attention_mask = peptide_sequence_encode.attention_mask.squeeze()

        batch = {
            "PDB_id": PDB_id,
            "peptide_sequence": peptide_sequence,
            "protein_sequence": protein_sequence,
            "text_sequence": text_sequence,
            "peptide_sequence_input_ids": peptide_sequence_input_ids,
            "peptide_sequence_attention_mask": peptide_sequence_attention_mask,
        }

        if self.text_tokenizer is not None:
            text_sequence_encode = self.text_tokenizer(text_sequence, truncation=True, max_length=self.text_max_sequence_len, padding='max_length', return_tensors='pt')
            text_sequence_input_ids = text_sequence_encode.input_ids
            text_sequence_attention_mask = text_sequence_encode.attention_mask
            batch["text_sequence_input_ids"] = text_sequence_input_ids
            batch["text_sequence_attention_mask"] = text_sequence_attention_mask

        return batch
    
    def __len__(self):
        return len(self.peptide_sequence_list)


def load_oracle_evaluator(device, input_model_path=None):
    if input_model_path is None:
        input_model_path = os.path.join("datasets_and_checkpoints/peptide_binding/MISATO/model_final.pth")

    eval_prediction_model = FoldingBindingInferenceModel(input_model_path=input_model_path)
    eval_prediction_model = eval_prediction_model.to(device)
    return eval_prediction_model


@torch.no_grad()
def evaluate_folding(protein_sequence_list, eval_prediction_model):
    eval_prediction_model.eval()

    PDB_data_list = []
    for protein_sequence in protein_sequence_list:
        print("protein_sequence", protein_sequence)

        PDB_data = eval_prediction_model.folding(protein_sequence)

        PDB_data_list.extend(PDB_data)

    return PDB_data_list


def get_aa_index(residue):
    letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12, "X":20}
    one_letter = three_to_one(residue)
    return letter_to_num[one_letter]


def parse_PDB_data(PDB_data):
    # https://www.biostat.jhsph.edu/~iruczins/teaching/260.655/links/pdbformat.pdf
    
    residue_type_list, ca_pos_list, c_pos_list, n_pos_list = [], [], [], []
    for line in PDB_data.split("\n"):
        if line.startswith("ATOM"):
            line = ' '.join(line.split())
            line = line.split(" ")
            atom_type = line[2]
            residue_type = line[3]
            x = float(line[6])
            y = float(line[7])
            z = float(line[8])
            if atom_type == "CA":
                residue_id = get_aa_index(residue_type)
                residue_type_list.append(residue_id)
                ca_pos_list.append([x, y, z])
            elif atom_type == "C":
                c_pos_list.append([x, y, z])
            elif atom_type == "N":
                n_pos_list.append([x, y, z])

    residue_type_list = torch.tensor(residue_type_list, dtype=torch.int64)
    ca_pos_list = torch.tensor(ca_pos_list, dtype=torch.float32)
    c_pos_list = torch.tensor(c_pos_list, dtype=torch.float32)
    n_pos_list = torch.tensor(n_pos_list, dtype=torch.float32)

    assert residue_type_list.shape[0] == ca_pos_list.shape[0] == c_pos_list.shape[0] == n_pos_list.shape[0]

    data = Data(
        x=residue_type_list,
        residue=residue_type_list,
        pos_ca=ca_pos_list,
        pos_c=c_pos_list,
        pos_n=n_pos_list,
    )
    return data


@torch.no_grad()
def evaluate_binding(peptide_PDB_id_list, PDB_id_list, peptide_idx2data, eval_prediction_model, device, args):
    from ProteinDT.datasets.dataset_MISATO import BatchMISATO
    
    eval_prediction_model.eval()

    result_list = []
    for peptide_PDB, PDB_id in zip(peptide_PDB_id_list, PDB_id_list):
        misato_data = peptide_idx2data[PDB_id]
        protein_batch = BatchMISATO.from_data_list([misato_data]).to(device)
            
        protein_residue = protein_batch.protein_residue
        protein_pos_N = protein_batch.protein_pos[protein_batch.protein_mask_n]
        protein_pos_Ca = protein_batch.protein_pos[protein_batch.protein_mask_ca]
        protein_pos_C = protein_batch.protein_pos[protein_batch.protein_mask_c]
        
        peptide_PDB_data = parse_PDB_data(peptide_PDB)
        peptide_batch = Batch.from_data_list([peptide_PDB_data]).to(device)

        peptide_residue = peptide_batch.residue
        peptide_pos_N = peptide_batch.pos_n
        peptide_pos_Ca = peptide_batch.pos_ca
        peptide_pos_C = peptide_batch.pos_c

        energy = eval_prediction_model.binding(
            protein_residue, protein_pos_N, protein_pos_Ca, protein_pos_C, protein_batch.protein_batch,
            peptide_residue, peptide_pos_N, peptide_pos_Ca, peptide_pos_C, peptide_batch.batch,
        )

        result_list.append(energy.detach().item())

    result_list = np.array(result_list)
    return result_list


@torch.no_grad()
def save_PDB_list(PDB_list, PDB_id_list, PDB_output_folder):
    for i, (PDB, PDB_id) in enumerate(zip(PDB_list, PDB_id_list)):
        file_name = os.path.join(PDB_output_folder, "{}_{}.txt".format(i, PDB_id))
        f = open(file_name, "w")
        f.write("".join(PDB))
    return
