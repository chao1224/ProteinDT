import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import re
import string

import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer, T5Tokenizer
from torch.utils.data import DataLoader

from ProteinDT.models import MultinomialDiffusion, T5Decoder, GaussianFacilitatorModel
from utils import slerp, ProteinDataset, ProteinSeqDataset, text_prompt_dict, load_oracle_evaluator, evaluate, analyze


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


@torch.no_grad()
def inference(dataloader, theta, text_condition_repr, device):
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader

    input_protein_sequence_list, edited_protein_sequence_list = [], []
    for batch_idx, batch in enumerate(L):
        protein_sequence_batch = batch["protein_sequence"]
        protein_sequence_input_ids = batch["protein_sequence_input_ids"].to(device)
        protein_sequence_attention_mask = batch["protein_sequence_attention_mask"].to(device)
        B = len(protein_sequence_batch)
        
        protein_output = CLAP_protein_model(protein_sequence_input_ids, protein_sequence_attention_mask)
        CLAP_protein_repr = protein_output["pooler_output"]
        CLAP_protein_repr = CLAP_protein2latent_model(CLAP_protein_repr)  # [B, hidden_dim]

        text_repr_batch = text_condition_repr.expand(B, -1)  # [B, hidden_dim]

        condition_repr = slerp(theta, CLAP_protein_repr, text_repr_batch)  # [B, hidden_dim]

        if args.AR_condition_mode == "aggregated":
            repeated_condition_repr = condition_repr.unsqueeze(1).expand(-1, args.num_repeat, -1)  # [B, num_repeat, hidden_dim]
            repeated_condition_repr = repeated_condition_repr.reshape(-1, args.condition_dim)  # [B*num_repeat, hidden_dim]

        else: # args.AR_condition_mode == "expanded":
            protein_output = CLAP_protein_model(protein_sequence_input_ids, protein_sequence_attention_mask)
            condition_repr = condition_repr.unsqueeze(1) # [B, 1, hidden_dim]
            CLAP_protein_token_repr = protein_output["last_hidden_state"]  # [B, max_seq_len, hidden_dim___]
            CLAP_protein_token_repr = CLAP_protein_token_repr[:, 1:, :]  # [B, max_seq_len-1, hidden_dim___]
            CLAP_protein_token_repr = CLAP_protein2latent_model(CLAP_protein_token_repr)  # [B, max_seq_len-1, hidden_dim]

            condition_repr = torch.concat([condition_repr, CLAP_protein_token_repr], dim=1)  # [B, max_seq_len, hidden_dim]
            repeated_condition_repr = condition_repr.unsqueeze(1).expand(-1, args.num_repeat, -1, -1)  # [B, num_repeat, max_seq_len, hidden_dim]
            assert args.protein_max_sequence_len == repeated_condition_repr.size()[2]
            repeated_condition_repr = repeated_condition_repr.reshape(-1, args.protein_max_sequence_len, args.condition_dim)  # [B*num_repeat, max_seq_len, hidden_dim]

        if args.decoder_distribution in ["T5Decoder"]:
            max_len = max(protein_sequence_attention_mask.sum(1)).item()
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
                num_beams = 10
            protein_sequence_pred_ids = decoder_distribution_model.inference(
                condition=repeated_condition_repr, protein_seq_attention_mask=None, max_seq_len=max_len,
                temperature=temperature, k=k, p=p, repetition_penalty=repetition_penalty, num_return_sequences=num_return_sequences, do_sample=do_sample, num_beams=num_beams
            )

        else:
            ##### add variable length
            min_len = min(protein_sequence_attention_mask.sum(1)).item()
            max_len = max(protein_sequence_attention_mask.sum(1)).item()
            protein_seq_attention_mask = torch.zeros((B*args.num_repeat, args.protein_max_sequence_len), device=device)
            for protein_seq_attention_mask_each in protein_seq_attention_mask:
                valid_length = random.randint(min_len, max_len)
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
        candidate_protein_sequence_list = protein_decoder_tokenizer.batch_decode(sequences=protein_sequence_pred_ids, skip_special_tokens=True)
        candidate_protein_sequence_list = [protein_sequence.replace(" ", "") for protein_sequence in candidate_protein_sequence_list]

        if args.oracle_mode == "protein":
            oracle_repr = CLAP_protein_repr
        elif args.oracle_mode == "text":
            oracle_repr = text_repr_batch
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

    parser.add_argument("--editing_task", type=str, default="BotNTB")
    parser.add_argument("--dataset_size", type=int, default=None)
    parser.add_argument("--text_prompt_id", type=int, default=101)
    parser.add_argument("--theta", type=float, default=0.01)
    parser.add_argument("--num_repeat", type=int, default=2)

    parser.add_argument("--pretrained_folder", type=str, default=None)
    parser.add_argument("--step_04_folder", type=str, default=None)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--output_text_file_path", type=str, default=None)

    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)

    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--condition_dim", type=int, default=256)

    parser.add_argument("--protein_backbone_model", type=str, default="ProtBERT_BFD", choices=["ProtBERT", "ProtBERT_BFD"])
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)
    parser.add_argument("--text_max_sequence_len", type=int, default=512)
    
    parser.add_argument("--facilitator_distribution", type=str, default="Gaussian", choices=["Gaussian"])

    parser.add_argument("--decoder_distribution", type=str, default="MultinomialDiffusion", choices=["T5Decoder", "MultinomialDiffusion"])
    
    ##### for AR #####
    parser.add_argument("--AR_generation_mode", type=str, default="01", choices=["01", "02"])
    parser.add_argument("--AR_condition_mode", type=str, default="aggregated", choices=["aggregated", "expanded"])

    ##### for SDE #####
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--beta_min", type=float, default=0.1)
    parser.add_argument("--beta_max", type=float, default=30)
    parser.add_argument("--num_diffusion_timesteps", type=int, default=1000)
    parser.add_argument("--SDE_type", type=str, default="VP")
    parser.add_argument("--score_network_type", type=str, default="BertProtBFD")
    parser.add_argument("--SDE_sampling_mode", type=str, default="simplified", choices=["simplified", "weighted"])

    ##### for post-selection #####
    parser.add_argument("--oracle_mode", type=str, default="protein", choices=["protein", "text", "condition"])

    args = parser.parse_args()
    print("arguments", args)
    assert args.pretrained_folder is not None
    assert args.step_04_folder is not None
    step_01_folder = args.pretrained_folder
    step_02_folder = os.path.join(args.pretrained_folder, "step_02_pairwise_representation")
    step_03_folder = os.path.join(args.pretrained_folder, "step_03_Gaussian_10")
    step_04_folder = args.step_04_folder

    assert args.SSL_emb_dim == args.condition_dim
    if args.AR_condition_mode == "expanded":
        assert args.decoder_distribution == "T5Decoder"

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
    
    if args.decoder_distribution in ["T5Decoder"]:
        protein_decoder_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False, chache_dir="../../data/temp_pretrained_t5_base")
    else:
        protein_decoder_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, chache_dir="../../data/temp_pretrained_PotBert")
    # print("protein_decoder_tokenizer pad_token_id", protein_decoder_tokenizer.pad_token_id)
    # print("protein_decoder_tokenizer sep_token_id", protein_decoder_tokenizer.sep_token_id)
    # print("protein_decoder_tokenizer eos_token_id", protein_decoder_tokenizer.eos_token_id)
    # print(CLAP_protein_tokenizer.get_vocab())
    # print(protein_decoder_tokenizer.get_vocab())

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
    decoder_distribution_model.to(device)
    decoder_distribution_model.eval()

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
    text_prompt_condition_repr = facilitator_distribution_model.inerence(text_repr=text_prompt_repr)  # [1, hidden_dim]

    ##### load protein sequence
    dataset_file_path = os.path.join(text_prompt_dict[args.editing_task]["data_folder"], "preprocessed_data.csv")
    dataset = ProteinDataset(
        dataset_file_path=dataset_file_path,
        dataset_size=args.dataset_size,
        protein_tokenizer=CLAP_protein_tokenizer,
        protein_max_sequence_len=args.protein_max_sequence_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    input_protein_sequence_list, edited_protein_sequence_list = inference(dataloader, args.theta, text_prompt_condition_repr, device)

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

    print("output: with theta {}".format(args.theta))
    output_dataset = ProteinSeqDataset(edited_protein_sequence_list, eval_protein_tokenizer, args.protein_max_sequence_len)
    output_dataloader = DataLoader(output_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    output_eval_list = evaluate(output_dataloader, eval_prediction_model, device, args).squeeze()
    file_path = os.path.join(args.output_folder, "{editing_task}_output_{theta}.png".format(editing_task=args.editing_task, theta=args.theta))
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
