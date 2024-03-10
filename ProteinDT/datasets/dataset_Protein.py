import os
import torch
from torch.utils.data import Dataset
import numpy as np


def load_file(file_path):
    f = open(file_path, 'r')
    seq_list = []
    for line_idx, line in enumerate(f.readlines()):
        seq = line.strip()
        seq_list.append(seq)
    return seq_list


class ProteinSequenceDataset(Dataset):
    def __init__(self, root, protein_tokenizer, protein_max_sequence_len):
        self.root = root

        protein_sequence_file = os.path.join(self.root, "protein_sequence.txt")
        self.protein_sequence_list = load_file(protein_sequence_file)
        print("len of protein_sequence {}".format(len(self.protein_sequence_list)))

        self.protein_tokenizer = protein_tokenizer
        self.protein_max_sequence_len = protein_max_sequence_len

        return

    def __getitem__(self, index):
        protein_sequence = self.protein_sequence_list[index]
        
        protein_sequence_encode = self.protein_tokenizer(protein_sequence, truncation=True, max_length=self.protein_max_sequence_len, padding='max_length', return_tensors='pt')
        protein_sequence_input_ids = protein_sequence_encode.input_ids.squeeze()
        protein_sequence_attention_mask = protein_sequence_encode.attention_mask.squeeze()

        batch = {
            "protein_sequence": protein_sequence,
            "protein_sequence_input_ids": protein_sequence_input_ids,
            "protein_sequence_attention_mask": protein_sequence_attention_mask,
        }

        return batch
    
    def __len__(self):
        return len(self.protein_sequence_list)
