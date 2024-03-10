import os
import re
import lmdb
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset


def pad_sequences(sequences, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)
        array = array.numpy()

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq
    return array


class SecondaryStructureDataset(Dataset):
    def __init__(self, data_file_path, protein_tokenizer, target='ss3'):
        self.protein_tokenizer = protein_tokenizer
        self.target = target
        self.ignore_index = -100

        env = lmdb.open(data_file_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))
        
        self.protein_sequence_list = []
        self.ss3_label_list = []
        self.ss8_label_list = []

        for index in range(num_examples):
            with env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
            # print(item.keys())
            protein_sequence = item["primary"]
            ss3_labels = item["ss3"]
            ss8_labels = item["ss8"]
            protein_length = item["protein_length"]

            if len(protein_sequence) > 1024:
                protein_sequence = protein_sequence[:1024]
                ss3_labels = ss3_labels[:1024]
                ss8_labels = ss8_labels[:1024]

            protein_sequence = re.sub(r"[UZOB]", "X", protein_sequence)
            protein_sequence = " ".join(protein_sequence)
                
            self.protein_sequence_list.append(protein_sequence)
            self.ss3_label_list.append(ss3_labels)
            self.ss8_label_list.append(ss8_labels)
        
        if self.target == "ss3":
            self.label_list = self.ss3_label_list
            self.num_labels = 3
        else:
            self.label_list = self.ss8_label_list
            self.num_labels = 8
        return

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        protein_sequence = self.protein_sequence_list[index]
        label = self.label_list[index]
        # protein_sequence_encode = self.protein_tokenizer(protein_sequence, truncation=True, max_length=self.protein_max_sequence_len, padding='max_length', return_tensors='pt')
        # protein_sequence_input_ids = protein_sequence_encode.input_ids#.squeeze()
        # protein_sequence_attention_mask = protein_sequence_encode.attention_mask#.squeeze()

        protein_sequence_encode = self.protein_tokenizer(list(protein_sequence), is_split_into_words=True, truncation=False, padding=True)
        protein_sequence_input_ids = np.array(protein_sequence_encode['input_ids'])
        protein_sequence_attention_mask = np.ones_like(protein_sequence_input_ids)

        label = np.asarray(label, np.int64)
        label = np.pad(label, (1, 1), 'constant', constant_values=self.ignore_index)

        # print("protein_sequence", len(protein_sequence))
        # print("protein_sequence_input_ids", len(protein_sequence_input_ids))
        # print("protein_sequence_attention_mask", len(protein_sequence_attention_mask))
        # print("label", len(label))

        return protein_sequence, protein_sequence_input_ids, protein_sequence_attention_mask, label

    def collate_fn(self, batch):
        protein_sequence, protein_sequence_input_ids, protein_sequence_attention_mask, label = tuple(zip(*batch))

        protein_sequence_input_ids = torch.from_numpy(pad_sequences(protein_sequence_input_ids, constant_value=self.protein_tokenizer.pad_token_id))
        protein_sequence_attention_mask = torch.from_numpy(pad_sequences(protein_sequence_attention_mask, constant_value=0))
        label = torch.from_numpy(pad_sequences(label, constant_value=self.ignore_index))

        output = {
            "protein_sequence": protein_sequence,
            "protein_sequence_input_ids": protein_sequence_input_ids,
            "protein_sequence_attention_mask": protein_sequence_attention_mask,
            "label": label,
        }
        return output
