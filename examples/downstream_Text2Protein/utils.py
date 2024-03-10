import re
import numpy as np
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, data_path, dataset_size, text_tokenizer, text_max_sequence_len):
        self.data_path = data_path
        self.dataset_size = dataset_size

        f = open(self.data_path, 'r')
        text_sequence_list = []
        for line in f.readlines():
            text_sequence = line.strip()
            text_sequence_list.append(text_sequence)
        self.text_sequence_list = text_sequence_list[:self.dataset_size]
        self.text_tokenizer = text_tokenizer
        self.text_max_sequence_len = text_max_sequence_len
        
        return

    def __getitem__(self, index):
        text_sequence = self.text_sequence_list[index]
        
        text_sequence_encode = self.text_tokenizer(text_sequence, truncation=True, max_length=self.text_max_sequence_len, padding='max_length', return_tensors='pt')
        text_sequence_input_ids = text_sequence_encode.input_ids.squeeze()
        text_sequence_attention_mask = text_sequence_encode.attention_mask.squeeze()

        batch = {
            "text_sequence": text_sequence,
            "text_sequence_input_ids": text_sequence_input_ids,
            "text_sequence_attention_mask": text_sequence_attention_mask,
        }

        return batch
    
    def __len__(self):
        return len(self.text_sequence_list)


class TextProteinPairDataset(Dataset):
    def __init__(self, data_path, protein_tokenizer, text_tokenizer, protein_max_sequence_len, text_max_sequence_len):
        self.data_path = data_path
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.protein_max_sequence_len = protein_max_sequence_len
        self.text_max_sequence_len = text_max_sequence_len

        text_sequence_list, protein_sequence_list = [], []
        f = open(self.data_path, 'r')
        for line_idx, line in enumerate(f.readlines()):
            line = line.strip()
            if line_idx % 2 == 0:
                text_sequence_list.append(line)
            else:
                line = re.sub(r"[UZOB]", "X", line)
                line = " ".join(line)
                protein_sequence_list.append(line)

        self.protein_sequence_list = protein_sequence_list
        self.text_sequence_list = text_sequence_list
        print("num of (protein-sequence, text) pair: {}".format(len(self.protein_sequence_list)))

        return

    def __getitem__(self, index):
        protein_sequence = self.protein_sequence_list[index]
        text_sequence = self.text_sequence_list[index]

        protein_sequence_encode = self.protein_tokenizer(protein_sequence, truncation=True, max_length=self.protein_max_sequence_len, padding='max_length', return_tensors='pt')
        protein_sequence_input_ids = protein_sequence_encode.input_ids.squeeze()
        protein_sequence_attention_mask = protein_sequence_encode.attention_mask.squeeze()

        text_sequence_encode = self.text_tokenizer(text_sequence, truncation=True, max_length=self.text_max_sequence_len, padding='max_length', return_tensors='pt')
        text_sequence_input_ids = text_sequence_encode.input_ids.squeeze()
        text_sequence_attention_mask = text_sequence_encode.attention_mask.squeeze()
        
        batch = {
            "protein_sequence": protein_sequence,
            "protein_sequence_input_ids": protein_sequence_input_ids,
            "protein_sequence_attention_mask": protein_sequence_attention_mask,
            "text_sequence": text_sequence,
            "text_sequence_input_ids": text_sequence_input_ids,
            "text_sequence_attention_mask": text_sequence_attention_mask,
        }

        return batch
    
    def __len__(self):
        return len(self.protein_sequence_list)


@torch.no_grad()
def evaluate(dataloader, model, evaluation_T_list, device):
    protein_repr_list, text_repr_list = [], []
    for batch in dataloader:
        protein_sequence_input_ids = batch["protein_sequence_input_ids"].to(device)
        protein_sequence_attention_mask = batch["protein_sequence_attention_mask"].to(device)
        text_sequence_input_ids = batch["text_sequence_input_ids"].to(device)
        text_sequence_attention_mask = batch["text_sequence_attention_mask"].to(device)
        
        protein_repr, text_repr = model(protein_sequence_input_ids, protein_sequence_attention_mask, text_sequence_input_ids, text_sequence_attention_mask)
        protein_repr = protein_repr.detach().cpu().numpy()
        text_repr = text_repr.detach().cpu().numpy()

        protein_repr_list.append(protein_repr)
        text_repr_list.append(text_repr)

    protein_repr_list = np.concatenate(protein_repr_list)
    text_repr_list = np.concatenate(text_repr_list)
    
    similarity_matrix = np.matmul(protein_repr_list, text_repr_list.T)
    N = similarity_matrix.shape[0]

    accuracy_list = []
    for evaluation_T in evaluation_T_list:
        accuracy = 0
        for i in range(N):
            start, end = i, i + evaluation_T
            similarity_matrix_segment = []
            for j in range(start, end):
                similarity_matrix_segment.append(similarity_matrix[i][j % N])
            optimal_index = np.argmax(similarity_matrix_segment)
            accuracy += (optimal_index == 0)
        accuracy = accuracy * 100. / N
        accuracy_list.append(accuracy)
    return accuracy_list
