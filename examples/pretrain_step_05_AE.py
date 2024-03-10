import os
import random
import numpy as np
import argparse
from tqdm import tqdm
import time

import torch
from torch import nn
import torch.optim as optim

from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader

from ProteinDT.datasets import ProteinSequenceDataset
from ProteinDT.models import RNNPrediction


def save_model(save_best):
    if save_best:
        global optimal_loss
        print("save model with loss: {:.5f}".format(optimal_loss))
        model_file = "model.pth"
        
        saved_file_path = os.path.join(step_05_folder, "AE_{}".format(model_file))
        torch.save(auto_encoding_model.state_dict(), saved_file_path)

    else:
        model_file = "model_final.pth"
        
        saved_file_path = os.path.join(step_05_folder, "AE_{}".format(model_file))
        torch.save(auto_encoding_model.state_dict(), saved_file_path)
    return


def train_AMP(dataloader):
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
            token_repr = CLIP_protein2latent_model(token_repr)  # [B, max_seq_len, SSL_emb_dim]

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

        if args.verbose and batch_idx % 100 == 0:
            print("CE Loss: {:.5f}".format(loss.item()))
            temp = logit.argmax(-1)
            print('temp', temp[:, :10])  # [B, max_seq_len]
            print("protein_sequence_input_ids", protein_sequence_input_ids[:, :10])

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
        save_model(save_best=True)
    print("SDE Loss: {:.5f}\tDecoding Loss: {:.5f}\tTime: {:.5f}".format(accum_AE_loss, accum_decoding_loss, time.time() - start_time))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay", type=float, default=0)

    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    
    parser.add_argument("--pretrained_folder", type=str, default=None)
    parser.add_argument("--output_folder", type=str, default=None)

    args = parser.parse_args()
    print("arguments", args)
    assert args.pretrained_folder is not None
    assert args.output_folder is not None
    step_01_folder = args.pretrained_folder
    step_02_folder = os.path.join(args.pretrained_folder, "step_02_pairwise_representation")
    step_05_folder = args.output_folder
    os.makedirs(step_05_folder, exist_ok=True)

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    CE_criterion = nn.CrossEntropyLoss()

    ##### Load pretrained protein model
    protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False, chache_dir="../data/temp_pretrained_ProtBert_BFD")
    protein_model = BertModel.from_pretrained("Rostlab/prot_bert_bfd", cache_dir="../data/temp_pretrained_ProtBert_BFD")
    input_model_path = os.path.join(args.pretrained_folder, "protein_model.pth")
    print("Loading protein model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    protein_model.load_state_dict(state_dict)
    protein_model.to(device)
    protein_model.eval()
    protein_dim = 1024

    ##### Load pretrained protein2latent model
    CLIP_protein2latent_model = nn.Linear(protein_dim, args.SSL_emb_dim)
    input_model_path = os.path.join(args.pretrained_folder, "protein2latent_model.pth")
    print("Loading protein2latent model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    CLIP_protein2latent_model.load_state_dict(state_dict)
    CLIP_protein2latent_model.to(device)
    CLIP_protein2latent_model.eval()

    for param in protein_model.parameters():
        param.requires_grad = False
    for param in CLIP_protein2latent_model.parameters():
        param.requires_grad = False
    
    # ### Add prediction head
    auto_encoding_model = RNNPrediction(args.SSL_emb_dim, protein_tokenizer.vocab_size)
    auto_encoding_model.to(device)

    model_param_group = [
        {"params": auto_encoding_model.parameters(), "lr": args.lr},
    ]
    optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
    optimal_loss = 1e10

    dataset = ProteinSequenceDataset(step_02_folder, protein_tokenizer=protein_tokenizer, protein_max_sequence_len=args.protein_max_sequence_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    for e in range(1, 1+args.epochs):
        print("Epoch {}".format(e))
        train_AMP(dataloader)
