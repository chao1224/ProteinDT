import numpy as np
import os
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer
from ProteinDT.TAPE_benchmark.models import BertForTokenClassification2, BertForSequenceClassification2
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')


text_prompt_dict = {
    "alpha": {
        101: "modify the amino acid sequence to have more alpha helices in the secondary structure",
        201: "modify the amino acid sequence to have fewer alpha helices in the secondary structure",
        "data_folder": "datasets_and_checkpoints/structure",
        "target_label": 0,
    },
    "beta": {
        101: "modify the amino acid sequence to have more beta sheets in the secondary structure",
        201: "modify the amino acid sequence to have fewer beta sheets in the secondary structure",
        "data_folder": "datasets_and_checkpoints/structure",
        "target_label": 1,
    },
    "Villin": {
        101: "modify the amino acid sequence to have higher stability",
        102: "modify the amino acid sequence to have fewer intrinsically disordered regions",
        201: "modify the amino acid sequence to have lower stability",
        202: "modify the amino acid sequence to have more intrinsically disordered regions",
        "data_folder": "datasets_and_checkpoints/stability/Villin",
    },
    "Pin1": {
        101: "modify the amino acid sequence to have higher stability",
        102: "modify the amino acid sequence to have fewer intrinsically disordered regions",
        201: "modify the amino acid sequence to have lower stability",
        202: "modify the amino acid sequence to have more intrinsically disordered regions",
        "data_folder": "datasets_and_checkpoints/stability/Pin1",
    },
    "hYAP65": {
        101: "modify the amino acid sequence to have higher stability",
        102: "modify the amino acid sequence to have fewer intrinsically disordered regions",
        201: "modify the amino acid sequence to have lower stability",
        202: "modify the amino acid sequence to have more intrinsically disordered regions",
        "data_folder": "datasets_and_checkpoints/stability/hYAP65",
    },
    "stability": {
        "data_folder": "datasets_and_checkpoints/stability_test"
    },
    "peptide_binding": {
        101: "modify the peptide amino acid sequence to have higher binding affinity with the target protein. The target protein satisfies the following property. {}",
        201: "modify the peptide amino acid sequence to have lower binding affinity with the target protein. The target protein satisfies the following property. {}",
        "data_folder": "datasets_and_checkpoints/peptide_binding/MISATO",
    },
}


class ProteinDataset(Dataset):
    def __init__(self, dataset_file_path, protein_tokenizer, protein_max_sequence_len, dataset_size=None):
        self.dataset_file_path = dataset_file_path
        self.dataset_size = dataset_size

        f = open(self.dataset_file_path, 'r')
        protein_sequence_list = []
        for line in f.readlines():
            line = line.strip().split(',')
            protein_sequence = line[0]
            protein_sequence = protein_sequence.replace(" ", "")
            protein_sequence = " ".join(protein_sequence)
            protein_sequence_list.append(protein_sequence)
        self.protein_sequence_list = protein_sequence_list
        if self.dataset_size is not None:
            self.protein_sequence_list = self.protein_sequence_list[:self.dataset_size]
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


class ProteinSeqDataset(Dataset):
    def __init__(self, protein_sequence_list, protein_tokenizer, protein_max_sequence_len):
        protein_sequence_list = [" ".join(seq) for seq in protein_sequence_list]
        self.protein_sequence_list = protein_sequence_list
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


def load_oracle_evaluator(editing_task, device, input_model_path=None):
    if editing_task in ["alpha", "beta"]:
        num_labels = 3
        eval_prediction_model = BertForTokenClassification2.from_pretrained(
            "Rostlab/prot_bert_bfd",
            mean_output=True,
            num_labels=num_labels,
        )
        if input_model_path is None:
            input_model_path = os.path.join("datasets_and_checkpoints/structure/oracle/pytorch_model_ss3.bin")

    elif editing_task in ["Villin", "Pin1", "hYAP65"]:
        num_labels = 1
        eval_prediction_model = BertForSequenceClassification2.from_pretrained(
            "Rostlab/prot_bert_bfd",
            mean_output=True,
            num_labels=num_labels,
        )
        if input_model_path is None:
            input_model_path = os.path.join("datasets_and_checkpoints/stability/oracle/pytorch_model_stability.bin")

    elif editing_task == "stability":
        # only for testing
        num_labels = 1
        eval_prediction_model = BertForSequenceClassification2.from_pretrained(
            "Rostlab/prot_bert_bfd",
            mean_output=True,
            num_labels=num_labels,
        )
        if input_model_path is None:
            input_model_path = os.path.join("datasets_and_checkpoints/stability/oracle/pytorch_model_stability.bin")

    print("Loading protein model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    eval_prediction_model.load_state_dict(state_dict)
    eval_prediction_model = eval_prediction_model.to(device)
    eval_protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
    return eval_prediction_model, eval_protein_tokenizer


def load_editing_dataset_and_loader(args, eval_protein_tokenizer):
    from torch.utils.data import DataLoader
    from ProteinDT.datasets import SecondaryStructureDataset, VillinDataset, Pin1Dataset, hYAP65Dataset, StabilityDataset

    if args.editing_task in ["alpha", "beta"]:
        test_dataset = SecondaryStructureDataset(data_file_path="./datasets_and_checkpoints/structure/secondary_structure_cb513.lmdb", protein_tokenizer=eval_protein_tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=test_dataset.collate_fn)
        criterion_eval = None

    elif args.editing_task == "Villin":
        test_dataset = VillinDataset(data_file_path="./datasets_and_checkpoints/stability/Villin/test_data.txt", protein_tokenizer=eval_protein_tokenizer, protein_max_sequence_len=args.protein_max_sequence_len)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size*2, shuffle=True, num_workers=args.num_workers)
        criterion_eval = nn.L1Loss(reduce=False)

    elif args.editing_task == "Pin1":
        test_dataset = Pin1Dataset(data_file_path="./datasets_and_checkpoints/stability/Pin1/test_data.txt", protein_tokenizer=eval_protein_tokenizer, protein_max_sequence_len=args.protein_max_sequence_len)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size*2, shuffle=True, num_workers=args.num_workers)
        criterion_eval = nn.L1Loss(reduce=False)

    elif args.editing_task == "hYAP65":
        test_dataset = hYAP65Dataset(data_file_path="./datasets_and_checkpoints/stability/hYAP65/test_data.txt", protein_tokenizer=eval_protein_tokenizer, protein_max_sequence_len=args.protein_max_sequence_len)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size*2, shuffle=True, num_workers=args.num_workers)
        criterion_eval = nn.L1Loss(reduce=False)

    elif args.editing_task == "stability":
        split_size = [0.8, 0.1, 0.1]
        test_dataset = StabilityDataset(root="../../data/stability/", seed=args.seed, mode="test", split_size=split_size, protein_tokenizer=eval_protein_tokenizer, protein_max_sequence_len=args.protein_max_sequence_len)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size*2, shuffle=True, num_workers=args.num_workers)
        criterion_eval = nn.L1Loss(reduce=False)

    return test_dataset, test_dataloader, criterion_eval


@torch.no_grad()
def evaluate(dataloader, eval_prediction_model, device, args):
    eval_prediction_model.eval()

    L = tqdm(dataloader)
    result_list = []
    for batch_id, batch in enumerate(L):
        protein_sequence = batch["protein_sequence"]
        protein_sequence_input_ids = batch["protein_sequence_input_ids"].to(device)
        protein_sequence_attention_mask = batch["protein_sequence_attention_mask"].to(device)
        
        output = eval_prediction_model(protein_sequence_input_ids, protein_sequence_attention_mask)
        logits = output.logits

        if args.editing_task in ["alpha", "beta"]:
            pred = logits.argmax(dim=-1, keepdim=False)
            pred = torch.where(protein_sequence_attention_mask==1, pred, -1)
            pred = (pred == text_prompt_dict[args.editing_task]["target_label"]).sum(-1)

        else:
            pred = logits

        result_list.append(pred.detach().cpu().numpy())

    result_list = np.concatenate(result_list)
    return result_list


def analyze(result_list, args, file_path=None):
    if args.editing_task in ["alpha", "beta"]:
        count = defaultdict(int)
        for c in result_list:
            count[c] += 1
        key_list = sorted(count.keys())
        for k in key_list:
            print(k, count[k])
    else:
        counts, bins = np.histogram(result_list, bins=20)
        
        plt.hist(bins[:-1], bins, weights=counts)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.clf()
    return


def slerp(theta, start, end):
    start_norm = start / torch.norm(start, dim=1, keepdim=True)
    end_norm = end / torch.norm(end, dim=1, keepdim=True)
    omega = torch.acos((start_norm*end_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-theta)*omega)/so).unsqueeze(1) * start + (torch.sin(theta*omega)/so).unsqueeze(1) * end
    return res


if __name__ == "__main__":
    start = torch.Tensor([[0, 0, 0], [1, 1, 1]])
    end = torch.Tensor([[1, 1, 1], [2, 2, 2]])
    
    start = torch.Tensor([[1, 1, 1], [2, 2, 2]])
    end = torch.Tensor([[3, 3, 3], [6, 6, 6]])
    
    start = torch.Tensor([[1, 1, 1], [1, 2, 2]])
    end = torch.Tensor([[2, 2, 2], [6, 6, 6]])

    theta_list = [0, 0.1, 0.5, 0.9, 1]
    for theta in theta_list:
        interpolation = slerp(theta, start, end)
        print(theta, interpolation)