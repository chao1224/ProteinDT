import os
import torch
from torch.utils.data import Dataset
import numpy as np


def load_file(file_path):
    uniprot_id2record_dict = {}
    f = open(file_path, 'r')
    seq_list = []
    for line_idx, line in enumerate(f.readlines()):
        seq = line.strip()
        seq_list.append(seq)
    return seq_list


class RepresentationPairDataset(Dataset):
    def __init__(self, root):
        self.root = root

        representation_file = os.path.join(self.root, "pairwise_representation.npz")
        representation_data = np.load(representation_file)
        print("representation_data", representation_data.keys)

        self.protein_repr_array = representation_data["protein_repr_array"]
        self.description_repr_array = representation_data["description_repr_array"]

        print("shape of protein_repr_array: {}".format(self.protein_repr_array.shape))
        print("shape of description_repr_array: {}".format(self.description_repr_array.shape))

        return

    def __getitem__(self, index):
        protein_repr = self.protein_repr_array[index]
        text_repr = self.description_repr_array[index]

        batch = {
            "protein_repr": protein_repr,
            "text_repr": text_repr,
        }

        return batch
    
    def __len__(self):
        return len(self.protein_repr_array)


class RepresentationPairWithRawDataDataset(Dataset):
    def __init__(self, root, prob_unconditional):
        self.root = root
        self.prob_unconditional = prob_unconditional

        representation_file = os.path.join(self.root, "empty_sequence.npz")
        representation_data = np.load(representation_file)
        self.empty_description = representation_data["protein_repr"][0]
        print("empty_description", self.empty_description.shape)

        representation_file = os.path.join(self.root, "pairwise_representation.npz")
        representation_data = np.load(representation_file)
        print("representation_data", representation_data.keys)

        self.protein_repr_array = representation_data["protein_repr_array"]
        self.description_repr_array = representation_data["description_repr_array"]

        print("shape of protein_repr_array: {}".format(self.protein_repr_array.shape))
        print("shape of description_repr_array: {}".format(self.description_repr_array.shape))

        protein_sequence_file = os.path.join(self.root, "protein_sequence.txt")
        text_sequence_file = os.path.join(self.root, "text_sequence.txt")

        self.protein_sequence_list = load_file(protein_sequence_file)
        print("len of protein_sequence {}".format(len(self.protein_sequence_list)))

        self.text_sequence_list = load_file(text_sequence_file)
        print("len of text_sequence {}".format(len(self.text_sequence_list)))

        return

    def __getitem__(self, index):
        protein_sequence = self.protein_sequence_list[index]
        text_sequence = self.text_sequence_list[index]
        protein_repr = self.protein_repr_array[index]
        text_repr = self.description_repr_array[index]

        if self.prob_unconditional > 0:
            roll_dice = np.random.uniform(0, 1)
            if roll_dice <= self.prob_unconditional:
                text_repr = self.empty_description
                protein_repr = self.empty_description
                text_sequence = ''
                protein_sequence = ''

        batch = {
            "protein_sequence": protein_sequence,
            "text_sequence": text_sequence,
            "protein_repr": protein_repr,
            "text_repr": text_repr,
        }

        return batch
    
    def __len__(self):
        return len(self.protein_repr_array)
