import os
import random
import argparse
import sys
import time
import re

from transformers import AutoTokenizer, OPTForCausalLM

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, BertModel, BertTokenizer
from utils import TextProteinPairDataset, evaluate
from ProteinDT.models import ProteinTextModel


def parse_Galatica_result(text_sequence, result):
    result = result.replace(text_sequence, "")
    result = result.split("[END_AMINO]")[0]
    return result


def load_model():
    assert args.pretrained_folder is not None
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ##### Load pretrained protein model
    if args.protein_backbone_model == "ProtBERT":
        protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, chache_dir="../../data/temp_pretrained_ProtBert")
        protein_model = BertModel.from_pretrained("Rostlab/prot_bert", cache_dir="../../data/temp_pretrained_ProtBert")
    elif args.protein_backbone_model == "ProtBERT_BFD":
        protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False, chache_dir="../../data/temp_pretrained_ProtBert_BFD")
        protein_model = BertModel.from_pretrained("Rostlab/prot_bert_bfd", cache_dir="../../data/temp_pretrained_ProtBert_BFD")
    protein_dim = 1024
    input_model_path = os.path.join(args.pretrained_folder, "protein_model.pth")
    print("Loading protein model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    protein_model.load_state_dict(state_dict)

    ##### Load pretrained text model
    text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../../data/temp_pretrained_SciBert")
    text_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../../data/temp_pretrained_SciBert")
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

    protein_model = protein_model.to(device)
    protein_model.eval()
    text_model = text_model.to(device)
    text_model.eval()
    protein2latent_model = protein2latent_model.to(device)
    protein2latent_model.eval()
    text2latent_model = text2latent_model.to(device)
    text2latent_model.eval()
    return protein_model, protein_tokenizer, text_model, text_tokenizer, protein2latent_model, text2latent_model


@torch.no_grad()
def extract_Galactica(num_repeat, protein_model, protein_tokenizer, text_model, text_tokenizer, protein2latent_model, text2latent_model):
    
    optimal_text_sequence_list, optimal_protein_sequence_list = [], []

    for idx, text_sequence in enumerate(text_sequence_list):
        print("idx: ", idx)
        output_file = "output/Galactica/{}.txt".format(idx)
        f = open(output_file, "r")
        lines = f.readlines()

        text_sequence = [lines[0]]
        text_sequence_encode = text_tokenizer(text_sequence, truncation=True, max_length=args.text_max_sequence_len, padding='max_length', return_tensors='pt')
        text_sequence_input_ids = text_sequence_encode.input_ids.squeeze(dim=1).to(device)
        text_sequence_attention_mask = text_sequence_encode.attention_mask.squeeze(dim=1).to(device)
        description_output = text_model(text_sequence_input_ids, text_sequence_attention_mask)
        text_repr = description_output["pooler_output"]
        text_repr = text2latent_model(text_repr)  # [1, hidden_dim]

        protein_sequence_list = []
        for line in lines[1:]:
            line = line.strip()
            
            protein_sequence = line.split(":")[1]
            protein_sequence = re.sub(r"[UZOB]", "X", protein_sequence)
            protein_sequence = " ".join(protein_sequence)
            protein_sequence_list.append(protein_sequence)
        protein_sequence_list = protein_sequence_list[:num_repeat]        

        protein_sequence_encode = protein_tokenizer(protein_sequence_list, truncation=True, max_length=args.protein_max_sequence_len, padding='max_length', return_tensors='pt')
        protein_sequence_input_ids = protein_sequence_encode.input_ids.to(device)
        protein_sequence_attention_mask = protein_sequence_encode.attention_mask.to(device)
        protein_output = protein_model(protein_sequence_input_ids, protein_sequence_attention_mask)
        protein_repr = protein_output["pooler_output"]
        protein_repr = protein2latent_model(protein_repr)

        print("text_repr", text_repr.shape)
        print("protein_repr", protein_repr.shape)

        similarity = torch.matmul(text_repr, protein_repr.transpose(0, 1))  # (num_repeat)
        optimal_id = torch.argmax(similarity)
        optimal_text_sequence_list.append(text_sequence)
        optimal_protein_sequence_list.append(protein_sequence_list[optimal_id])

    return optimal_text_sequence_list, optimal_protein_sequence_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dataset_size", type=int, default=200)

    parser.add_argument("--input_text_file_path", type=str, default="step_01_text_retrieval.txt")
    parser.add_argument("--num_repeat", type=int, default=32)

    parser.add_argument("--pretrained_folder", type=str, default="../../output/ProteinDT/ProtBERT_BFD-512-1e-5-1-text-512-1e-5-1-EBM_NCE-1-batch-9-gpu-8-epoch-10")
    parser.add_argument("--protein_backbone_model", type=str, default="ProtBERT_BFD", choices=["ProtBERT", "ProtBERT_BFD"])
    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)
    parser.add_argument("--text_max_sequence_len", type=int, default=512)
    
    args = parser.parse_args()
    print("arguments", args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    f = open(args.input_text_file_path, 'r')
    text_sequence_list = []
    for line in f.readlines():
        text_sequence = line.strip()
        text_sequence_list.append(text_sequence)
    text_sequence_list = text_sequence_list[:args.dataset_size]
    
    ##### generate and save protein sequence #####
    galactica_tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-1.3b")
    galactica_model = OPTForCausalLM.from_pretrained("facebook/galactica-1.3b", device_map="auto")
    galactica_num_return_sequences = 4

    for idx, text_sequence in enumerate(text_sequence_list):
        print("idx: ", idx)
        output_file = "output/Galactica/{}.txt".format(idx)
        if os.path.isfile(output_file):
            continue

        f = open(output_file, "w")
        
        print(text_sequence, file=f)

        text_sequence = "{} Question: Can you generate one similar protein amino acid sequences satisfying (just amino acids and keywords or references)? [START_AMINO]".format(text_sequence)
        text_ids = galactica_tokenizer(text_sequence, return_tensors="pt").input_ids.to("cuda")
        
        count = 0
        for _ in range(args.num_repeat // galactica_num_return_sequences):
            outputs = galactica_model.generate(
                text_ids,
                max_new_tokens=args.protein_max_sequence_len,
                do_sample=True,
                top_p=0.95,
                temperature=1.0,
                use_cache=True,
                top_k=5,
                repetition_penalty=1.0,
                length_penalty=1,
                num_return_sequences=galactica_num_return_sequences,
            )
            
            for i in range(galactica_num_return_sequences):
                protein_sequence_answer = galactica_tokenizer.decode(outputs[i])
                protein_sequence_answer = parse_Galatica_result(text_sequence, protein_sequence_answer)
                print("{}: {}".format(count, protein_sequence_answer), file=f)
                count += 1
        
        f.flush()
        f.close()

    protein_model, protein_tokenizer, text_model, text_tokenizer, protein2latent_model, text2latent_model = load_model()

    ###### extract #####
    num_repeat = 16
    optimal_text_sequence_list, optimal_protein_sequence_list = extract_Galactica(
        num_repeat=num_repeat,
        protein_model=protein_model, protein_tokenizer=protein_tokenizer,
        text_model=text_model, text_tokenizer=text_tokenizer,
        protein2latent_model=protein2latent_model, text2latent_model=text2latent_model)

    output_text_file_path = "output/Galactica/step_02_inference.txt"
    if output_text_file_path is not None:
        f = open(output_text_file_path, 'w')
        for text_sequence, protein_sequence in zip(optimal_text_sequence_list, optimal_protein_sequence_list):
            print(text_sequence, file=f)
            print(protein_sequence, file=f)
        f.flush()
        f.close()

    # ###### evaluate #####
    model = ProteinTextModel(protein_model, text_model, protein2latent_model, text2latent_model)
    model.eval()
    model.to(device)

    dataset = TextProteinPairDataset(
        data_path=output_text_file_path,
        protein_tokenizer=protein_tokenizer,
        text_tokenizer=text_tokenizer,
        protein_max_sequence_len=args.protein_max_sequence_len,
        text_max_sequence_len=args.text_max_sequence_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    evaluation_T_list = [4, 10, 20]
    accuracy_list = evaluate(dataloader, model, evaluation_T_list, device)
    for evaluation_T, accuracy in zip(evaluation_T_list, accuracy_list):
        print("evaluation_T: {}\taccuracy: {}".format(evaluation_T, accuracy))
