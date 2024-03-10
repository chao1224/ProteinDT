import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import time
import re

import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer, T5Tokenizer
from torch.utils.data import DataLoader

from utils import TextDataset, TextProteinPairDataset, evaluate
from ProteinDT.models import MultinomialDiffusion, T5Decoder, GaussianFacilitatorModel, ProteinTextModel


@torch.no_grad()
def inference(dataloader):
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader

    optimal_text_sequence_list, optimal_protein_sequence_list = [], []
    for batch_idx, batch in enumerate(L):
        text_sequence = batch["text_sequence"]
        text_sequence_input_ids = batch["text_sequence_input_ids"].to(device)
        text_sequence_attention_mask = batch["text_sequence_attention_mask"].to(device)
        optimal_text_sequence_list.extend(text_sequence)
        B = len(text_sequence)
        
        description_output = text_model(text_sequence_input_ids, text_sequence_attention_mask)
        text_repr = description_output["pooler_output"]
        text_repr = text2latent_model(text_repr)  # [B, hidden_dim]

        if args.use_facilitator:
            condition_repr = facilitator_distribution_model.inerence(text_repr=text_repr)  # [B, hidden_dim]
        else:
            condition_repr = text_repr

        repeated_condition_repr = condition_repr.unsqueeze(1).expand(-1, args.num_repeat, -1)  # [B, num_repeat, hidden_dim]
        repeated_condition_repr = repeated_condition_repr.reshape(-1, args.condition_dim)  # [B*num_repeat, hidden_dim]

        if args.decoder_distribution in ["T5Decoder"]:
            if args.AR_generation_mode == "01":
                temperature = 1.0
                k = 40
                p = 0.9
                repetition_penalty = 1.0
                num_return_sequences = 1
                do_sample = True
                num_beams = 1
            elif args.AR_generation_mode == "02":
                temperature = 1.0
                k = 40
                p = 0.9
                repetition_penalty = 1.0
                num_return_sequences = 1
                do_sample = False
                num_beams = 1
            protein_sequence_pred_ids = decoder_distribution_model.inference(
                condition=repeated_condition_repr, protein_seq_attention_mask=None, max_seq_len=args.protein_max_sequence_len,
                temperature=temperature, k=k, p=p, repetition_penalty=repetition_penalty, num_return_sequences=num_return_sequences, do_sample=do_sample, num_beams=num_beams
            )

        else:
            ##### add variable length
            protein_seq_attention_mask = torch.zeros((B*args.num_repeat, args.protein_max_sequence_len), device=device)
            for protein_seq_attention_mask_each in protein_seq_attention_mask:
                valid_length = random.randint(300, args.protein_max_sequence_len)
                protein_seq_attention_mask_each[:valid_length] = 1
            protein_seq_attention_mask = protein_seq_attention_mask.bool()

            protein_sequence_output = decoder_distribution_model.inference(
                condition=repeated_condition_repr, protein_seq_attention_mask=protein_seq_attention_mask, max_seq_len=args.protein_max_sequence_len, mode=args.SDE_sampling_mode)
            protein_sequence_pred_ids = torch.argmax(protein_sequence_output, dim=-1)  # (B*num_repeat, max_seq_len)

            ##### truncate after index
            for protein_sequence_pred_id in protein_sequence_pred_ids:
                index = None
                for pid, pred_id in enumerate(protein_sequence_pred_id):
                    if pred_id != protein_decoder_tokenizer.pad_token_id:
                        continue
                    index = pid
                    break
                if index is not None:
                    protein_sequence_pred_id[index:] = protein_decoder_tokenizer.pad_token_id

        ##### clean-ups
        protein_sequence_list = protein_decoder_tokenizer.batch_decode(sequences=protein_sequence_pred_ids, skip_special_tokens=True)
        protein_sequence_list = [protein_sequence.replace(" ", "") for protein_sequence in protein_sequence_list]

        ##### get protein representation
        cleaned_protein_sequence_list = []
        for protein_sequence in protein_sequence_list:
            protein_sequence = re.sub(r"[UZOB]", "X", protein_sequence)
            protein_sequence = " ".join(protein_sequence)
            cleaned_protein_sequence_list.append(protein_sequence)
        protein_sequence_encode = protein_tokenizer(cleaned_protein_sequence_list, truncation=True, max_length=args.protein_max_sequence_len, padding='max_length', return_tensors='pt')
        protein_sequence_input_ids = protein_sequence_encode.input_ids.to(device)
        protein_sequence_attention_mask = protein_sequence_encode.attention_mask.to(device)
        protein_output = protein_model(protein_sequence_input_ids, protein_sequence_attention_mask)
        protein_repr = protein_output["pooler_output"]
        protein_repr = protein2latent_model(protein_repr)

        ##### select the most similar protein sequence using ProteinCLIP
        for text_id in range(B):
            start, end = text_id * args.num_repeat, (text_id + 1) * args.num_repeat
            text_repr_segment = text_repr[text_id]
            protein_repr_segment = protein_repr[start:end]
            similarity = torch.matmul(text_repr_segment, protein_repr_segment.transpose(0, 1))  # (num_repeat)
            optimal_id = torch.argmax(similarity)
            optimal_id = start + optimal_id
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
    parser.add_argument("--pretrained_folder", type=str, default=None)
    parser.add_argument("--step_04_folder", type=str, default=None)
    parser.add_argument("--output_text_file_path", type=str, default=None)

    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    parser.add_argument("--num_repeat", type=int, default=2)

    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--condition_dim", type=int, default=256)

    parser.add_argument("--protein_backbone_model", type=str, default="ProtBERT_BFD", choices=["ProtBERT", "ProtBERT_BFD"])
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)
    parser.add_argument("--text_max_sequence_len", type=int, default=512)
    
    parser.add_argument("--facilitator_distribution", type=str, default="Gaussian", choices=["Gaussian"])
    parser.add_argument("--use_facilitator", dest="use_facilitator", action="store_true")
    parser.add_argument("--no_use_facilitator", dest="use_facilitator", action="store_false")
    parser.set_defaults(use_facilitator=True)

    parser.add_argument("--decoder_distribution", type=str, default="T5Decoder", choices=["T5Decoder", "MultinomialDiffusion"])
    
    ##### for AR #####
    parser.add_argument("--AR_generation_mode", type=str, default="01", choices=["01", "02"])
    
    ##### for SDE #####
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--beta_min", type=float, default=0.1)
    parser.add_argument("--beta_max", type=float, default=30)
    parser.add_argument("--num_diffusion_timesteps", type=int, default=1000)
    parser.add_argument("--SDE_type", type=str, default="VP")
    parser.add_argument("--score_network_type", type=str, default="BertProtBFD")
    parser.add_argument("--SDE_sampling_mode", type=str, default="simplified", choices=["simplified", "weighted"])

    args = parser.parse_args()
    print("arguments", args)
    assert args.pretrained_folder is not None
    assert args.step_04_folder is not None
    step_01_folder = args.pretrained_folder
    step_02_folder = os.path.join(args.pretrained_folder, "step_02_pairwise_representation")
    step_03_folder = os.path.join(args.pretrained_folder, "step_03_Gaussian_10")
    step_04_folder = args.step_04_folder

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

    ##### Load pretrained facilitator model
    if args.facilitator_distribution == "Gaussian":
        facilitator_distribution_model = GaussianFacilitatorModel(args.SSL_emb_dim)
        # TODO: will check later.
        input_model_path = os.path.join(step_03_folder, "facilitator_distribution_model.pth")
        print("Loading facilitator_distribution model from {}...".format(input_model_path))
        state_dict = torch.load(input_model_path, map_location='cpu')
        facilitator_distribution_model.load_state_dict(state_dict)
    
    if args.decoder_distribution in ["T5Decoder"]:
        protein_decoder_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False, chache_dir="../../data/temp_pretrained_t5_base")
    else:
        protein_decoder_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, chache_dir="../../data/temp_pretrained_ProtBert")
    print("protein_decoder_tokenizer pad_token_id", protein_decoder_tokenizer.pad_token_id)
    print("protein_decoder_tokenizer sep_token_id", protein_decoder_tokenizer.sep_token_id)
    print("protein_decoder_tokenizer eos_token_id", protein_decoder_tokenizer.eos_token_id)

    ##### Load pretrained decoder model
    if args.decoder_distribution == "MultinomialDiffusion":
        mask_id = 4
        decoder_distribution_model = MultinomialDiffusion(
            hidden_dim=args.hidden_dim,
            condition_dim=args.condition_dim, mask_id=mask_id,
            beta_min=args.beta_min, beta_max=args.beta_max, num_diffusion_timesteps=args.num_diffusion_timesteps,
            num_classes=protein_decoder_tokenizer.vocab_size, score_network_type=args.score_network_type)
    elif args.decoder_distribution == "T5Decoder":
        decoder_distribution_model = T5Decoder(
            hidden_dim=args.condition_dim,
            tokenizer=protein_decoder_tokenizer,
            T5_model=args.score_network_type)

    model_file_path = os.path.join(step_04_folder, "decoder_distribution_model.pth")
    print("Loading decoder_distribution model from {}...".format(model_file_path))
    state_dict = torch.load(model_file_path, map_location='cpu')
    decoder_distribution_model.load_state_dict(state_dict)

    protein_model = protein_model.to(device)
    protein_model.eval()
    text_model = text_model.to(device)
    text_model.eval()
    protein2latent_model = protein2latent_model.to(device)
    protein2latent_model.eval()
    text2latent_model = text2latent_model.to(device)
    text2latent_model.eval()
    facilitator_distribution_model = facilitator_distribution_model.to(device)
    facilitator_distribution_model.eval()
    decoder_distribution_model.to(device)
    decoder_distribution_model.eval()

    dataset = TextDataset(
        data_path=args.input_text_file_path,
        dataset_size=args.dataset_size,
        text_tokenizer=text_tokenizer,
        text_max_sequence_len=args.text_max_sequence_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimal_text_sequence_list, optimal_protein_sequence_list = inference(dataloader)

    if args.output_text_file_path is not None:
        f = open(args.output_text_file_path, 'w')
        for text_sequence, protein_sequence in zip(optimal_text_sequence_list, optimal_protein_sequence_list):
            print(text_sequence, file=f)
            print(protein_sequence, file=f)
        f.flush()
        f.close()

    ########## evaluate
    model = ProteinTextModel(protein_model, text_model, protein2latent_model, text2latent_model)
    model.eval()
    model.to(device)

    dataset = TextProteinPairDataset(
        data_path=args.output_text_file_path,
        protein_tokenizer=protein_tokenizer,
        text_tokenizer=text_tokenizer,
        protein_max_sequence_len=args.protein_max_sequence_len,
        text_max_sequence_len=args.text_max_sequence_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    evaluation_T_list = [4, 10, 20]
    accuracy_list = evaluate(dataloader, model, evaluation_T_list, device)
    for evaluation_T, accuracy in zip(evaluation_T_list, accuracy_list):
        print("evaluation_T: {}\taccuracy: {}".format(evaluation_T, accuracy))
