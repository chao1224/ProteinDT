import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import re
import string
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
from torch.utils.data import DataLoader

from ProteinDT.models import GaussianFacilitatorModel, RNNPrediction
from utils import ProteinDataset, ProteinSeqDataset, text_prompt_dict, load_oracle_evaluator, evaluate, analyze


def finetune_AE(dataloader, protein_model, CLAP_protein2latent_model, protein_tokenizer, optimizer, CE_criterion):
    scaler = torch.cuda.amp.GradScaler()
    auto_encoding_model.train()
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader

    start_time = time.time()
    accum_AE_loss, accum_decoding_loss = 0, 0
    for batch_idx, batch in enumerate(L):
        protein_sequence_input_ids = batch["protein_sequence_input_ids"].to(device)
        protein_sequence_attention_mask = batch["protein_sequence_attention_mask"].to(device)
        
        with torch.no_grad():
            protein_output = protein_model(protein_sequence_input_ids, protein_sequence_attention_mask)
            token_repr = protein_output["last_hidden_state"]  # [B, max_seq_len, 1024]
            token_repr = CLAP_protein2latent_model(token_repr)  # [B, max_seq_len, SSL_emb_dim]

        with torch.cuda.amp.autocast():
            logit = auto_encoding_model(token_repr)  # [B, max_seq_len, vocab_size]

            target_protein_seq_input_ids = protein_sequence_input_ids # [B, max_sequence_len]
            target_protein_seq_attention_mask = protein_sequence_attention_mask # [B, max_sequence_len]
            flattened_logits = torch.flatten(logit, start_dim=0, end_dim=1)  # [B * max_sequence_len, vocab_size]
            flattened_ids = torch.flatten(target_protein_seq_input_ids, start_dim=0, end_dim=1)  # [B * max_sequence_len]
            flattened_mask = torch.flatten(target_protein_seq_attention_mask, start_dim=0, end_dim=1)  # [B * max_sequence_len]
            total_loss = CE_criterion(flattened_logits, flattened_ids)  # [B * max_sequence_len]
            masked_loss = total_loss * flattened_mask  # [B * max_sequence_len]
            total_loss = torch.mean(total_loss)
            masked_loss = masked_loss.sum() / flattened_mask.sum()

            loss = (total_loss + masked_loss) / 2

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        accum_AE_loss += loss.item()

    accum_AE_loss /= len(L)
    accum_decoding_loss /= len(L)
    global optimal_loss
    temp_loss = accum_AE_loss
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
    print("SDE Loss: {:.5f}\tDecoding Loss: {:.5f}\tTime: {:.5f}".format(accum_AE_loss, accum_decoding_loss, time.time() - start_time))
    return


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


@torch.no_grad()
def select_protein_list(candidate_protein_sequence_list, oracle_repr, B, protein_sequence_batch):
    assert B * args.num_repeat == len(candidate_protein_sequence_list)
    assert oracle_repr.size()[0] == B

    input_protein_sequence_list, edited_protein_sequence_list = [], []
    ##### get protein representation
    parsed_protein_sequence_list = []
    for protein_sequence in candidate_protein_sequence_list:
        protein_sequence = re.sub(r"[UZOB]", "X", protein_sequence)
        protein_sequence = " ".join(protein_sequence)
        parsed_protein_sequence_list.append(protein_sequence)
    protein_sequence_encode = CLAP_protein_tokenizer(parsed_protein_sequence_list, truncation=True, max_length=args.protein_max_sequence_len, padding='max_length', return_tensors='pt')
    protein_sequence_input_ids = protein_sequence_encode.input_ids.to(device)
    protein_sequence_attention_mask = protein_sequence_encode.attention_mask.to(device)
    protein_output = CLAP_protein_model(protein_sequence_input_ids, protein_sequence_attention_mask)
    protein_repr_test = protein_output["pooler_output"]
    protein_repr_test = CLAP_protein2latent_model(protein_repr_test)
    assert protein_repr_test.size()[0] == B * args.num_repeat

    ##### select the sequences most similar to the input protein sequences
    for condition_id in range(B):
        start, end = condition_id * args.num_repeat, (condition_id + 1) * args.num_repeat
        oracle_protein_repr = oracle_repr[condition_id]  # hidden_dim
        candidate_protein_repr = protein_repr_test[start:end]  # num_repeat, hidden_dim

        similarity = torch.matmul(oracle_protein_repr, candidate_protein_repr.transpose(0, 1))  # (num_repeat)
        optimal_id = torch.argmax(similarity)
        optimal_id = start + optimal_id
        input_protein_sequence_list.append(protein_sequence_batch[condition_id].replace(" ", ""))
        edited_protein_sequence_list.append(candidate_protein_sequence_list[optimal_id])
    return input_protein_sequence_list, edited_protein_sequence_list


def latent_optimization(protein_sequence_input_ids, latent_code_init, text_CLAP_repr):
    repeated_latent_code_init = latent_code_init.unsqueeze(0).expand(args.num_repeat, -1, -1, -1)  # [B, num_repeat, max_seq_len, hidden_dim]
    repeated_latent_code_init = repeated_latent_code_init.flatten(0, 1)  # [num_repeat*B, max_seq_len, hidden_dim]

    repeated_protein_sequence_input_ids = protein_sequence_input_ids.unsqueeze(1).expand(-1, args.num_repeat, -1)  # [B, num_repeat]
    repeated_protein_sequence_input_ids = repeated_protein_sequence_input_ids.flatten(0, 1)  # [num_repeat*B, max_seq_len]

    random_noise = torch.randn(repeated_latent_code_init.size()).to(device)  / args.temperature # [num_repeat*B, max_seq_len, hidden_dim]
    latent_code = repeated_latent_code_init.detach().clone() + random_noise
    latent_code.requires_grad = True
    optimizer = optim.Adam([latent_code], lr=args.lr)

    if args.verbose:
        current_L = tqdm(range(args.epochs))
    else:
        current_L = range(args.epochs)
        
    for i in current_L:
        t = i / args.epochs
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        protein_repr = CLAP_protein_model.pooler(latent_code)  # [num_repeat*B, hidden_dim]
        protein_repr = CLAP_protein2latent_model(protein_repr)  # [num_repeat*B, hidden_dim]

        loss_01 =  ((protein_repr - text_CLAP_repr) ** 2).mean()
        loss_02 =  ((repeated_latent_code_init - latent_code) ** 2).mean()

        loss = (args.lambda_value) * loss_01 + (1 - args.lambda_value) * loss_02

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    latent_code = latent_code.detach()

    # TODO
    latent_code = CLAP_protein2latent_model(latent_code)  # [num_repeat*B, max_seq_len, hidden_dim]
    protein_sequence_output = auto_encoding_model(latent_code)  # [num_repeat*B, max_seq_len, vocab_size]
    return protein_sequence_output


def inference(dataloader, text_prompt_CLAP_repr, device):
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader

    input_protein_sequence_list, edited_protein_sequence_list = [], []
    for batch_idx, batch in enumerate(L):
        protein_sequence_batch = batch["protein_sequence"]
        protein_sequence_input_ids = batch["protein_sequence_input_ids"].to(device)
        protein_sequence_attention_mask = batch["protein_sequence_attention_mask"].to(device)
        # print("protein_sequence_batch", protein_sequence_batch)
        
        B = protein_sequence_input_ids.size()[0]
        text_CLAP_repr = text_prompt_CLAP_repr.expand(B*args.num_repeat, -1)  # [B, hidden_dim]
        text_CLAP_repr = text_CLAP_repr.detach()

        with torch.no_grad():
            protein_output = CLAP_protein_model(protein_sequence_input_ids, protein_sequence_attention_mask)
        latent_code_init = protein_output["last_hidden_state"]  # [B, max_seq_len, hidden_dim]
        protein_CLAP_repr = protein_output["pooler_output"]
        protein_CLAP_repr = CLAP_protein2latent_model(protein_CLAP_repr)  # [B, hidden_dim]

        protein_sequence_output = latent_optimization(protein_sequence_input_ids, latent_code_init, text_CLAP_repr)
        protein_sequence_output = protein_sequence_output.detach()

        protein_sequence_pred_ids = torch.argmax(protein_sequence_output, dim=-1)  # (B*num_repeat, max_seq_len)
        
        # ##### truncate after index
        # for protein_sequence_pred_id in protein_sequence_pred_ids:
        #     index = None
        #     for pid, pred_id in enumerate(protein_sequence_pred_id):
        #         if pred_id != protein_decoder_tokenizer.pad_token_id:
        #             continue
        #         index = pid
        #         break
        #     if index is not None:
        #         protein_sequence_pred_id[index:] = protein_decoder_tokenizer.pad_token_id

        ##### clean-ups
        candidate_protein_sequence_list = protein_decoder_tokenizer.batch_decode(sequences=protein_sequence_pred_ids, skip_special_tokens=True)
        candidate_protein_sequence_list = [protein_sequence.replace(" ", "") for protein_sequence in candidate_protein_sequence_list]
        print("candidate_protein_sequence_list", len(candidate_protein_sequence_list))
        # print("candidate_protein_sequence_list", candidate_protein_sequence_list)

        if args.oracle_mode == "protein":
            oracle_repr = protein_CLAP_repr
        elif args.oracle_mode == "text":
            oracle_repr = text_prompt_CLAP_repr.expand(B, -1)  # [B, hidden_dim]
        input_protein_sequence_list_, edited_protein_sequence_list_ = select_protein_list(
            candidate_protein_sequence_list=candidate_protein_sequence_list, oracle_repr=oracle_repr, B=B, protein_sequence_batch=protein_sequence_batch)
        input_protein_sequence_list.extend(input_protein_sequence_list_)
        edited_protein_sequence_list.extend(edited_protein_sequence_list_)

    return input_protein_sequence_list, edited_protein_sequence_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.5)

    parser.add_argument("--AE_lr", type=float, default=5e-4)
    parser.add_argument("--AE_epochs", type=int, default=50)
    parser.add_argument("--AE_decay", type=float, default=0)

    parser.add_argument("--editing_task", type=str, default="Villin")
    parser.add_argument("--dataset_size", type=int, default=None)
    parser.add_argument("--text_prompt_id", type=int, default=101)
    parser.add_argument("--lambda_value", type=float, default=0.1)
    parser.add_argument("--num_repeat", type=int, default=2)

    parser.add_argument("--pretrained_folder", type=str, default=None)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--output_text_file_path", type=str, default=None)
    parser.add_argument("--step_05_folder", type=str, default="step_05_AE_test")

    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)

    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--condition_dim", type=int, default=256)

    parser.add_argument("--protein_backbone_model", type=str, default="ProtBERT_BFD", choices=["ProtBERT", "ProtBERT_BFD"])
    parser.add_argument("--protein_max_sequence_len", type=int, default=512) #50
    parser.add_argument("--text_max_sequence_len", type=int, default=512) #50
    
    parser.add_argument("--facilitator_distribution", type=str, default="Gaussian", choices=["Gaussian"])    

    ##### for post-selection #####
    parser.add_argument("--oracle_mode", type=str, default="protein", choices=["protein", "text"])

    args = parser.parse_args()
    print("arguments", args)
    assert args.pretrained_folder is not None
    assert args.step_05_folder is not None
    step_01_folder = args.pretrained_folder
    step_02_folder = os.path.join(args.pretrained_folder, "step_02_pairwise_representation")
    step_03_folder = os.path.join(args.pretrained_folder, "step_03_Gaussian_10")
    step_05_folder = args.step_05_folder

    assert args.SSL_emb_dim == args.condition_dim

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
        CLAP_protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        CLAP_protein_model = BertModel.from_pretrained("Rostlab/prot_bert")
    elif args.protein_backbone_model == "ProtBERT_BFD":
        CLAP_protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
        CLAP_protein_model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
    protein_dim = 1024
    input_model_path = os.path.join(args.pretrained_folder, "protein_model.pth")
    print("Loading protein model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    CLAP_protein_model.load_state_dict(state_dict)

    ##### Load pretrained text model
    CLAP_text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../../data/temp_pretrained_SciBert")
    CLAP_text_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../../data/temp_pretrained_SciBert")
    text_dim  = 768
    input_model_path = os.path.join(args.pretrained_folder, "text_model.pth")
    print("Loading text model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    CLAP_text_model.load_state_dict(state_dict)

    ##### Load pretrained protein2latent model
    CLAP_protein2latent_model = nn.Linear(protein_dim, args.SSL_emb_dim)
    input_model_path = os.path.join(args.pretrained_folder, "protein2latent_model.pth")
    print("Loading protein2latent model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    CLAP_protein2latent_model.load_state_dict(state_dict)

    ##### Load pretrained text2latent model
    CLAP_text2latent_model = nn.Linear(text_dim, args.SSL_emb_dim)
    input_model_path = os.path.join(args.pretrained_folder, "text2latent_model.pth")
    print("Loading text2latent model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    CLAP_text2latent_model.load_state_dict(state_dict)

    ##### Load pretrained facilitator model
    if args.facilitator_distribution == "Gaussian":
        facilitator_distribution_model = GaussianFacilitatorModel(args.SSL_emb_dim)
        input_model_path = os.path.join(step_03_folder, "facilitator_distribution_model.pth")
        print("Loading facilitator_distribution model from {}...".format(input_model_path))
        state_dict = torch.load(input_model_path, map_location='cpu')
        facilitator_distribution_model.load_state_dict(state_dict)

    protein_decoder_tokenizer = CLAP_protein_tokenizer
    print("protein_decoder_tokenizer pad_token_id", protein_decoder_tokenizer.pad_token_id)
    print("protein_decoder_tokenizer sep_token_id", protein_decoder_tokenizer.sep_token_id)
    print("protein_decoder_tokenizer eos_token_id", protein_decoder_tokenizer.eos_token_id)
    
    ##### Load prediction head
    auto_encoding_model = nn.Linear(args.SSL_emb_dim, CLAP_protein_tokenizer.vocab_size)
    auto_encoding_model = RNNPrediction(args.SSL_emb_dim, CLAP_protein_tokenizer.vocab_size)
    input_model_path = os.path.join(step_05_folder, "AE_model.pth")
    print("Loading auto_encoding_model model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    auto_encoding_model.load_state_dict(state_dict)

    CLAP_protein_model = CLAP_protein_model.to(device)
    CLAP_protein_model.eval()
    CLAP_text_model = CLAP_text_model.to(device)
    CLAP_text_model.eval()
    CLAP_protein2latent_model = CLAP_protein2latent_model.to(device)
    CLAP_protein2latent_model.eval()
    CLAP_text2latent_model = CLAP_text2latent_model.to(device)
    CLAP_text2latent_model.eval()
    facilitator_distribution_model = facilitator_distribution_model.to(device)
    facilitator_distribution_model.eval()
    auto_encoding_model = auto_encoding_model.to(device)
    auto_encoding_model.eval()
    
    ##### obtain text sequence representation
    text_prompt = text_prompt_dict[args.editing_task][args.text_prompt_id]
    print("===== the text prompt is : {} =====".format(text_prompt))
    text_prompt_sequence_list = [text_prompt]
    text_prompt_sequence_encode = CLAP_text_tokenizer(text_prompt_sequence_list, truncation=True, max_length=args.text_max_sequence_len, padding='max_length', return_tensors='pt')
    text_prompt_sequence_input_ids = text_prompt_sequence_encode.input_ids.to(device)
    text_prompt_sequence_attention_mask = text_prompt_sequence_encode.attention_mask.to(device)
    description_output = CLAP_text_model(text_prompt_sequence_input_ids, text_prompt_sequence_attention_mask)
    text_prompt_repr = description_output["pooler_output"]
    text_prompt_repr = CLAP_text2latent_model(text_prompt_repr)  # [1, hidden_dim]
    text_prompt_CLAP_repr = facilitator_distribution_model.inerence(text_repr=text_prompt_repr)  # [1, hidden_dim]

    ##### load protein sequence
    dataset_file_path = os.path.join(text_prompt_dict[args.editing_task]["data_folder"], "preprocessed_data.csv")
    dataset = ProteinDataset(
        dataset_file_path=dataset_file_path,
        dataset_size=args.dataset_size,
        protein_tokenizer=CLAP_protein_tokenizer,
        protein_max_sequence_len=args.protein_max_sequence_len)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=args.num_workers)
    
    optimal_loss = 1e10
    auto_encoding_model.train()
    model_param_group = [{"params": auto_encoding_model.parameters(), "lr": args.AE_lr}]
    CE_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_param_group, weight_decay=args.AE_decay)
    optimal_loss = 1e10
    print("========== start finetuning AE ==========")
    for _ in range(args.AE_epochs):
        finetune_AE(dataloader, CLAP_protein_model, CLAP_protein2latent_model, CLAP_protein_tokenizer, optimizer, CE_criterion)
    print("========== done finetuning AE ==========")
    auto_encoding_model.eval()
    print("\n\n\n")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    input_protein_sequence_list, edited_protein_sequence_list = inference(dataloader, text_prompt_CLAP_repr, device)

    if args.output_folder is None:
        exit()
    
    ##### Load pretrained model
    eval_prediction_model, eval_protein_tokenizer = load_oracle_evaluator(args.editing_task, device)

    print('input:')
    input_dataset = ProteinSeqDataset(input_protein_sequence_list, eval_protein_tokenizer, args.protein_max_sequence_len)
    input_dataloader = DataLoader(input_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    input_eval_list = evaluate(input_dataloader, eval_prediction_model, device, args).squeeze()
    file_path = os.path.join(args.output_folder, "{editing_task}_input.png".format(editing_task=args.editing_task))
    analyze(input_eval_list, args, file_path)

    print("output: with lambda_value {}".format(args.lambda_value))
    output_dataset = ProteinSeqDataset(edited_protein_sequence_list, eval_protein_tokenizer, args.protein_max_sequence_len)
    output_dataloader = DataLoader(output_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    output_eval_list = evaluate(output_dataloader, eval_prediction_model, device, args).squeeze()
    file_path = os.path.join(args.output_folder, "{editing_task}_output_{lambda_value}.png".format(editing_task=args.editing_task, lambda_value=args.lambda_value))
    analyze(output_eval_list, args, file_path)

    if args.output_text_file_path is None:
        args.output_text_file_path = os.path.join(args.output_folder, "step_01_editing.txt")

    f = open(args.output_text_file_path, 'w')
    hit, total = 0, 0
    for input_protein, input_eval, edited_protein, output_eval in zip(input_protein_sequence_list, input_eval_list, edited_protein_sequence_list, output_eval_list):
        print('input,{},{}'.format(input_protein, input_eval), file=f)
        print('output,{},{}'.format(edited_protein, output_eval), file=f)

        total += 1
        
        if args.editing_task in ["alpha", "beta"]:
            if args.text_prompt_id in [101]:
                if output_eval > input_eval:
                    hit += 1
            elif args.text_prompt_id in [201]:
                if output_eval < input_eval:
                    hit += 1

        elif args.editing_task in ["Villin", "Pin1", "hYAP65"]:
            if args.text_prompt_id in [101, 102]:
                if output_eval < input_eval:
                    hit += 1
            elif args.text_prompt_id in [201, 202]:
                if output_eval > input_eval:
                    hit += 1

    hit_ratio = 100. * hit / total
    print("hit: {}".format(hit))
    print("total: {}".format(total))
    print("hit ratio: {}".format(hit_ratio))
