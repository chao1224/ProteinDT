import os
import random
import numpy as np
import argparse
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader

from ProteinDT.models import ProteinTextModel
from ProteinDT.datasets import SwissProtCLAPDataset


@torch.no_grad()
def extract(dataloader):
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    
    protein_repr_list, description_repr_list, protein_seq_list, text_seq_list = [], [], [], []
    for batch_idx, batch in enumerate(L):
        protein_seq = batch["protein_sequence"]
        text_seq = batch["text_sequence"]
        protein_sequence_input_ids = batch["protein_sequence_input_ids"].to(device)
        protein_sequence_attention_mask = batch["protein_sequence_attention_mask"].to(device)
        text_sequence_input_ids = batch["text_sequence_input_ids"].to(device)
        text_sequence_attention_mask = batch["text_sequence_attention_mask"].to(device)
        
        protein_repr, description_repr = model(protein_sequence_input_ids, protein_sequence_attention_mask, text_sequence_input_ids, text_sequence_attention_mask)
        protein_repr_list.append(protein_repr.detach().cpu().numpy())
        description_repr_list.append(description_repr.detach().cpu().numpy())
        protein_seq_list.extend(protein_seq)
        text_seq_list.extend(text_seq)

    protein_repr_array = np.concatenate(protein_repr_list)
    description_repr_array= np.concatenate(description_repr_list)
    return protein_repr_array, description_repr_array, protein_seq_list, text_seq_list


@torch.no_grad()
def extract_AMP(dataloader):
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    
    protein_repr_list, description_repr_list, protein_seq_list, text_seq_list = [], [], [], []
    for batch_idx, batch in enumerate(L):
        protein_seq = batch["protein_sequence"]
        text_seq = batch["text_sequence"]
        protein_sequence_input_ids = batch["protein_sequence_input_ids"].to(device)
        protein_sequence_attention_mask = batch["protein_sequence_attention_mask"].to(device)
        text_sequence_input_ids = batch["text_sequence_input_ids"].to(device)
        text_sequence_attention_mask = batch["text_sequence_attention_mask"].to(device)
        
        with torch.cuda.amp.autocast():
            protein_repr, description_repr = model(protein_sequence_input_ids, protein_sequence_attention_mask, text_sequence_input_ids, text_sequence_attention_mask)
        protein_repr_list.append(protein_repr.detach().cpu().numpy())
        description_repr_list.append(description_repr.detach().cpu().numpy())
        protein_seq_list.extend(protein_seq)
        text_seq_list.extend(text_seq)

    protein_repr_array = np.concatenate(protein_repr_list)
    description_repr_array= np.concatenate(description_repr_list)
    return protein_repr_array, description_repr_array, protein_seq_list, text_seq_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--protein_backbone_model", type=str, default="ProtBERT_BFD", choices=["ProtBERT", "ProtBERT_BFD"])
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)
    parser.add_argument("--text_max_sequence_len", type=int, default=512)
    
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    
    parser.add_argument("--use_AMP", dest="use_AMP", action="store_true")
    parser.add_argument("--no_AMP", dest="use_AMP", action="store_false")
    parser.set_defaults(use_AMP=True)

    parser.add_argument("--pretrained_folder", type=str, default=None)

    args = parser.parse_args()
    print("arguments", args)

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ##### Load pretrained protein model
    if args.protein_backbone_model == "ProtBERT":
        protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, chache_dir="../data/temp_pretrained_ProtBert")
        protein_model = BertModel.from_pretrained("Rostlab/prot_bert", cache_dir="../data/temp_pretrained_ProtBert")
    elif args.protein_backbone_model == "ProtBERT_BFD":
        protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False, chache_dir="../data/temp_pretrained_ProtBert_BFD")
        protein_model = BertModel.from_pretrained("Rostlab/prot_bert_bfd", cache_dir="../data/temp_pretrained_ProtBert_BFD")
    protein_dim = 1024
    input_model_path = os.path.join(args.pretrained_folder, "protein_model.pth")
    print("Loading protein model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    protein_model.load_state_dict(state_dict)

    ##### Load pretrained text model
    text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../data/temp_pretrained_SciBert")
    text_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../data/temp_pretrained_SciBert")
    text_dim  = 768
    input_model_path = os.path.join(args.pretrained_folder, "text_model.pth")
    print("Loading text model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    text_model.load_state_dict(state_dict)

    ##### Load pretrained protein2latent model
    protein2latent_model = nn.Linear(protein_dim, args.SSL_emb_dim)
    input_model_path = os.path.join(args.pretrained_folder, "protein2latent_model.pth")
    print("Loading protein2latent model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    protein2latent_model.load_state_dict(state_dict)

    ##### Load pretrained text2latent model
    text2latent_model = nn.Linear(text_dim, args.SSL_emb_dim)
    input_model_path = os.path.join(args.pretrained_folder, "text2latent_model.pth")
    print("Loading text2latent model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    text2latent_model.load_state_dict(state_dict)

    model = ProteinTextModel(protein_model, text_model, protein2latent_model, text2latent_model)
    model.eval()
    model.to(device)

    text_sequence = ""
    text_sequence_encode = text_tokenizer(text_sequence, truncation=True, max_length=args.text_max_sequence_len, padding='max_length', return_tensors='pt')
    text_sequence_input_ids = text_sequence_encode.input_ids.to(device)
    text_sequence_attention_mask = text_sequence_encode.attention_mask.to(device)

    description_output = text_model(text_sequence_input_ids, text_sequence_attention_mask)
    description_repr = description_output["pooler_output"]
    description_repr = text2latent_model(description_repr)
    description_repr = description_repr.detach().cpu().numpy()
    print("description_repr", description_repr.shape)

    protein_sequence = ""
    protein_sequence_encode = protein_tokenizer(protein_sequence, truncation=True, max_length=args.protein_max_sequence_len, padding='max_length', return_tensors='pt')
    protein_sequence_input_ids = protein_sequence_encode.input_ids.to(device)
    protein_sequence_attention_mask = protein_sequence_encode.attention_mask.to(device)
    protein_output = protein_model(protein_sequence_input_ids, protein_sequence_attention_mask)
    protein_repr = protein_output["pooler_output"]
    protein_repr = protein2latent_model(protein_repr)
    protein_repr = protein_repr.detach().cpu().numpy()
    print("protein_repr", protein_repr.shape)

    assert args.pretrained_folder is not None
    output_folder = os.path.join(args.pretrained_folder, "step_02_pairwise_representation")
    os.makedirs(output_folder, exist_ok=True)

    saved_file_path = os.path.join(output_folder, "empty_sequence")
    np.savez(saved_file_path, description_repr=description_repr, protein_repr=protein_repr)
