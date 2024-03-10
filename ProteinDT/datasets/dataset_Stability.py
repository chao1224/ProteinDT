import os
import re
from torch.utils.data import Dataset
from sklearn.utils import shuffle


class StabilityDataset(Dataset):
    def __init__(self, root, seed, mode, split_size, protein_tokenizer, protein_max_sequence_len):
        self.labels, self.seqs = [], []
        
        for p in ['1', '2', '3', '4']:
            file_path = os.path.join(root, 'rd{}_stability_scores'.format(p))

            with open(file_path) as f:
                lines = f.readlines()[1:]
            for line in lines:
                items = line.split("\t")
                seq, score = items[1], items[-1][:-2]
                if '_' not in seq and score != '':
                    seq = re.sub(r"[UZOB]", "X", seq)
                    seq = " ".join(seq)
                    score = float(score)
                    self.seqs.append(seq)
                    self.labels.append(score)
        
        self.seqs, self.labels = shuffle(self.seqs, self.labels, random_state=seed)
        
        train_size = int(split_size[0] * len(self.labels))
        train_val_size = int((split_size[0] + split_size[1]) * len(self.labels))
        if mode == "train":
            self.seqs = self.seqs[:train_size]
            self.labels = self.labels[:train_size]
        elif mode == "val":
            self.seqs = self.seqs[train_size:train_val_size]
            self.labels = self.labels[train_size:train_val_size]
        elif mode == "test":
            self.seqs = self.seqs[train_val_size:]
            self.labels = self.labels[train_val_size:]
        else:
            raise ValueError('Invalid split mode')

        self.protein_tokenizer = protein_tokenizer
        self.protein_max_sequence_len = protein_max_sequence_len
        return
                 
    def __getitem__(self, index):
        protein_sequence = self.seqs[index]
        label = self.labels[index]
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
        return len(self.labels)


if __name__ == "__main__":
    protein_dataset = StabilityDataset(root="../../data/stability/", seed=100, subset=False, mode="train", split_size=[0.8, 0.1, 0.1])
    print("Total number of seq: ", len(protein_dataset.labels))
    protein_dataset = StabilityDataset(root="../../data/stability/", seed=100, subset=False, mode="val", split_size=[0.8, 0.1, 0.1])
    print("Total number of seq: ", len(protein_dataset.labels))
    protein_dataset = StabilityDataset(root="../../data/stability/", seed=100, subset=False, mode="test", split_size=[0.8, 0.1, 0.1])
    print("Total number of seq: ", len(protein_dataset.labels))
