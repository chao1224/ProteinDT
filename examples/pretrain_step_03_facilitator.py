import os
import random
import numpy as np
import argparse
from tqdm import tqdm
import time

import torch
import torch.optim as optim

from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader

from ProteinDT.datasets import RepresentationPairDataset
from ProteinDT.models import GaussianPriorModel


def save_model(save_best):    
    if save_best:
        global optimal_loss
        print("save model with loss: {:.5f}".format(optimal_loss))
        model_file = "model.pth"
        
        saved_file_path = os.path.join(step_03_folder, "facilitator_distribution_{}".format(model_file))
        torch.save(facilitator_distribution_model.state_dict(), saved_file_path)

    else:
        model_file = "model_final.pth"
        
        saved_file_path = os.path.join(step_03_folder, "facilitator_distribution_{}".format(model_file))
        torch.save(facilitator_distribution_model.state_dict(), saved_file_path)

    return


def train(dataloader):
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    
    start_time = time.time()
    accum_loss = 0
    for batch_idx, batch in enumerate(L):
        protein_repr = batch["protein_repr"].to(device)
        text_repr = batch["text_repr"].to(device)
        
        loss = facilitator_distribution_model(protein_repr=protein_repr, text_repr=text_repr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accum_loss += loss.item()

    accum_loss /= len(L)
    global optimal_loss
    temp_loss = accum_loss
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
    print("Loss: {:.5f}\tTime: {:.5f}".format(accum_loss, time.time() - start_time))
    return


def train_AMP(dataloader):
    scaler = torch.cuda.amp.GradScaler()

    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    
    start_time = time.time()
    accum_loss = 0
    for batch_idx, batch in enumerate(L):
        protein_repr = batch["protein_repr"].to(device)
        text_repr = batch["text_repr"].to(device)
        
        with torch.cuda.amp.autocast():
            loss = facilitator_distribution_model(protein_repr=protein_repr, text_repr=text_repr)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        accum_loss += loss.item()
        
    accum_loss /= len(L)
    global optimal_loss
    temp_loss = accum_loss
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
    print("Loss: {:.5f}\tTime: {:.5f}".format(accum_loss, time.time() - start_time))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--protein_backbone_model", type=str, default="ProtBERT_BFD", choices=["ProtBERT", "ProtBERT_BFD"])
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)
    parser.add_argument("--text_max_sequence_len", type=int, default=512)
    parser.add_argument("--protein_lr", type=float, default=1e-5)
    parser.add_argument("--protein_lr_scale", type=float, default=1e-1)
    parser.add_argument("--text_lr", type=float, default=1e-5)
    parser.add_argument("--text_lr_scale", type=float, default=1e-1)
    parser.add_argument("--CL_neg_samples", type=int, default=1)
    parser.add_argument("--CL_loss", type=str, default="EBM_NCE")
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument("--decay", type=float, default=0)

    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=False)

    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    
    parser.add_argument("--use_AMP", dest="use_AMP", action="store_true")
    parser.add_argument("--no_AMP", dest="use_AMP", action="store_false")
    parser.set_defaults(use_AMP=True)

    parser.add_argument("--facilitator_distribution", type=str, default="Gaussian", choices=["Gaussian"])
    parser.add_argument("--pretrained_folder", type=str, default=None)
    parser.add_argument("--output_model_folder", type=str, default=None)

    args = parser.parse_args()
    print("arguments", args)
    assert args.pretrained_folder is not None
    assert args.output_model_folder is not None

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if args.facilitator_distribution == "Gaussian":
        facilitator_distribution_model = GaussianPriorModel(args.SSL_emb_dim)
    facilitator_distribution_model.train()
    facilitator_distribution_model.to(device)

    model_param_group = [
        {"params": facilitator_distribution_model.parameters(), "lr": args.protein_lr * args.protein_lr_scale},
    ]
    optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
    optimal_loss = 1e10

    step_02_folder = os.path.join(args.pretrained_folder, "step_02_pairwise_representation")
    dataset = RepresentationPairDataset(step_02_folder)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    step_03_folder = args.output_model_folder
    os.makedirs(step_03_folder, exist_ok=True)

    for e in range(1, args.epochs+1):
        print("Epoch {}".format(e))
        if args.use_AMP:
            train_AMP(dataloader)
        else:
            train(dataloader)
    