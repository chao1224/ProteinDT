import os
import torch
from torch.utils.data import Dataset


def load_protein_sequence_or_text_sequence_file(file_path):
    uniprot_id2record_dict = {}
    f = open(file_path, 'r')
    for line_idx, line in enumerate(f.readlines()):
        line = line.strip()
        if line_idx % 2 == 0:
            uniprot = line
        else:
            protein_sequence = line
            uniprot_id2record_dict[uniprot] = protein_sequence
    return uniprot_id2record_dict


class SwissProtCLAPDataset(Dataset):
    def __init__(self, root, protein_tokenizer, text_tokenizer, protein_max_sequence_len, text_max_sequence_len):
        self.root = root
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.protein_max_sequence_len = protein_max_sequence_len
        self.text_max_sequence_len = text_max_sequence_len

        protein_sequence_file = os.path.join(root, "protein_sequence.txt")
        text_sequence_file = os.path.join(root, "text_sequence.txt")

        uniprot_id2protein_sequence = load_protein_sequence_or_text_sequence_file(protein_sequence_file)
        # print("len of protein_sequence {}".format(len(uniprot_id2protein_sequence)))

        uniprot_id2text_sequence = load_protein_sequence_or_text_sequence_file(text_sequence_file)
        # print("len of protein_sequence {}".format(len(uniprot_id2text_sequence)))

        protein_sequence_list, text_sequence_list = [], []
        for uniprot_id, protein_sequence in uniprot_id2protein_sequence.items():
            if len(protein_sequence) > self.protein_max_sequence_len:
                continue
            text_sequence = uniprot_id2text_sequence[uniprot_id]

            # already done in preprocessing
            # protein_sequence = re.sub(r"[UZOB]", "X", protein_sequence)
            protein_sequence = " ".join(protein_sequence)
            protein_sequence_list.append(protein_sequence)
            text_sequence_list.append(text_sequence)
        
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


if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
    from torch.utils.data import DataLoader

    protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, chache_dir="../../data/temp_pretrained_ProtBert")

    text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../../data/temp_pretrained_SciBert")

    dataset = SwissProtCLAPDataset(
        root="../../data/SwissProtCLAP",
        protein_tokenizer=protein_tokenizer,
        text_tokenizer=text_tokenizer,
        protein_max_sequence_len=512,
        text_max_sequence_len=512
    )
    print("len of dataset", len(dataset))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    for batch in dataloader:
        protein_sequence_list = batch["protein_sequence"]
        text_sequence_list = batch["text_sequence"] 
        for protein_sequence, text_sequence in zip(protein_sequence_list, text_sequence_list):
            print(protein_sequence.replace(" ", ""))
            print(text_sequence)
            print()
        break
