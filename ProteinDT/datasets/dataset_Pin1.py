import os
import re
from torch.utils.data import Dataset
from sklearn.utils import shuffle


class Pin1Dataset(Dataset):
    def __init__(self, data_file_path, protein_tokenizer, protein_max_sequence_len):
        self.labels, self.seqs = [], []
        
        self.protein_sequnece_list, self.label_list = [], []
        f = open(data_file_path, 'r')
        for line in f.readlines():
            line = line.strip()
            line = line.split(",")
            
            protein_sequence = line[0]
            assert len(protein_sequence) == 39
            protein_sequence = re.sub(r"[UZOB]", "X", protein_sequence)
            protein_sequence = " ".join(protein_sequence)

            label = float(line[1])
            self.protein_sequnece_list.append(protein_sequence)
            self.label_list.append(label)

        self.protein_tokenizer = protein_tokenizer
        self.protein_max_sequence_len = protein_max_sequence_len
        return
                 
    def __getitem__(self, index):
        protein_sequence = self.protein_sequnece_list[index]
        label = self.label_list[index]
        protein_sequence_encode = self.protein_tokenizer(protein_sequence, truncation=True, max_length=self.protein_max_sequence_len, padding='max_length', return_tensors='pt')
        protein_sequence_input_ids = protein_sequence_encode.input_ids.squeeze()
        protein_sequence_attention_mask = protein_sequence_encode.attention_mask.squeeze()

        output = {
            "protein_sequence": protein_sequence,
            "protein_sequence_input_ids": protein_sequence_input_ids,
            "protein_sequence_attention_mask": protein_sequence_attention_mask,
            "label": label,
        }
        return output

    def __len__(self):
        return len(self.label_list)
