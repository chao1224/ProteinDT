import openai
import os
import random
import argparse
import sys
import time
import re
openai.api_key = ""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, BertModel, BertTokenizer, AutoTokenizer
from utils import TextProteinPairDataset, evaluate
from ProteinDT.models import ProteinTextModel


def complete_chatgpt(messages):
    received = False
    temperature = 0
    while not received:
        try:
            response = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo",
                # model="gpt-3.5-turbo-16k",
                model="gpt-3.5-turbo-0301",
                messages=messages,
                temperature=temperature,
                frequency_penalty=0.2,
                n=None)
            raw_generated_text = response["choices"][0]["message"]['content']   
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt error.\n\n")
                print("prompt too long")
                return "prompt too long"
            if error == AssertionError:
                print("Assert error:", sys.exc_info()[1])
            else:
                print("API error:", error)
            time.sleep(1)
    return raw_generated_text


def load_model():
    assert args.pretrained_folder is not None
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ##### Load pretrained protein model
    if args.protein_backbone_model == "ProtBERT":
        protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, chache_dir="../../data/temp_pretrained_PotBert")
        protein_model = BertModel.from_pretrained("Rostlab/prot_bert", cache_dir="../../data/temp_pretrained_PotBert")
    elif args.protein_backbone_model == "ProtBERT_BFD":
        protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False, chache_dir="../../data/temp_pretrained_PotBert_BFD")
        protein_model = BertModel.from_pretrained("Rostlab/prot_bert_bfd", cache_dir="../../data/temp_pretrained_PotBert_BFD")
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
def extract_ChatGPT(num_repeat, protein_model, protein_tokenizer, text_model, text_tokenizer, protein2latent_model, text2latent_model):
    pattern = re.compile('[A-Z]{5,}')

    optimal_text_sequence_list, optimal_protein_sequence_list = [], []

    for idx, text_sequence in enumerate(text_sequence_list):
        print("idx: ", idx)
        output_file = "output/ChatGPT/{}.txt".format(idx)
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
            parsed_results = pattern.findall(line)
            if len(parsed_results) == 0:
                continue
            protein_sequence = parsed_results[0]
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

    ##### generate and save protein sequence #####
    f = open(args.input_text_file_path, 'r')
    text_sequence_list = []
    for line in f.readlines():
        text_sequence = line.strip()
        text_sequence_list.append(text_sequence)
    text_sequence_list = text_sequence_list[:args.dataset_size]

    for idx, text_sequence in enumerate(text_sequence_list):
        output_file = "output/ChatGPT/{}.txt".format(idx)
        if os.path.isfile(output_file):
            continue

        f = open(output_file, "w")

        text_sequence = "Can you generate one protein amino acid sequences satisfying the following property (just AA sequence, and no explanation)? {}".format(text_sequence)
        messages = [
            {"role": "system", "content": "You are an expert in protein design."},
            {"role": "user", "content": text_sequence},
        ]
        protein_sequence_answer = complete_chatgpt(messages)
        messages.append({"role": "assistant", "content": protein_sequence_answer})
        print(text_sequence, file=f)
        print("=== answer 0 ===", file=f)
        print(protein_sequence_answer, file=f)

        for idx in range(1, args.num_repeat):
            print("=== answer {} ===".format(idx), file=f)
            messages.append({"role": "user", "content": "Can you give me one more protein sequence that is different from the previous (just AA sequence, and no explanation)?"})
            protein_sequence_answer = complete_chatgpt(messages)
            messages.append({"role": "assistant", "content": protein_sequence_answer})
            print(protein_sequence_answer, file=f)
            if protein_sequence_answer == "prompt too long":
                break

        f.flush()
        f.close()

    protein_model, protein_tokenizer, text_model, text_tokenizer, protein2latent_model, text2latent_model = load_model()

    ###### extract #####
    num_repeat = 16
    optimal_text_sequence_list, optimal_protein_sequence_list = extract_ChatGPT(
        num_repeat=num_repeat,
        protein_model=protein_model, protein_tokenizer=protein_tokenizer,
        text_model=text_model, text_tokenizer=text_tokenizer,
        protein2latent_model=protein2latent_model, text2latent_model=text2latent_model)

    output_text_file_path = "output/ChatGPT/step_02_inference.txt"
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
    