import os
import random
import argparse
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from ProteinDT.datasets import MISATODataset, MISATODataLoader
from ProteinDT.models import BindingModel


def save_model(save_best):
    if not args.output_model_dir == "":
        if save_best:
            print("save model with optimal loss")
            output_model_path = os.path.join(args.output_model_dir, "model.pth")
            saved_model_dict = {}
            saved_model_dict["binding_model"] = binding_model.state_dict()
            torch.save(saved_model_dict, output_model_path)

        else:
            print("save model in the last epoch")
            output_model_path = os.path.join(args.output_model_dir, "model_final.pth")
            saved_model_dict = {}
            saved_model_dict["binding_model"] = binding_model.state_dict()
            torch.save(saved_model_dict, output_model_path)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    # for CDConv
    parser.add_argument("--CDConv_radius", type=float, default=4)
    parser.add_argument("--CDConv_kernel_size", type=int, default=21)
    parser.add_argument("--CDConv_kernel_channels", type=int, nargs="+", default=[24])
    parser.add_argument("--CDConv_geometric_raddi_coeff", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--CDConv_channels", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--CDConv_base_width", type=int, default=64)

    parser.add_argument("--loss", type=str, default="MAE", choices=["MSE", "MAE"])
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="None")
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--print_every_epoch", type=int, default=5)
    parser.add_argument("--output_model_dir", type=str, default="")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--no_verbose", dest="verbose", action="store_false")
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    print("args", args)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    data_folder = "../../data/MISATO"
    dataset = MISATODataset(data_folder)
    loader = MISATODataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    binding_model = BindingModel(args).to(device)
    binding_model.train()

    if args.loss == "MSE":
        criterion = nn.MSELoss(reduction="mean")
    elif args.loss == "MAE":
        criterion = nn.L1Loss(reduction="mean")

    # set up optimizer
    model_param_group = [
        {"params": binding_model.parameters(), "lr": args.lr},
    ]
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model_param_group, lr=args.lr, weight_decay=args.decay)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    for e in range(args.epochs):
        if args.verbose:
            L = tqdm(loader)
        else:
            L = loader

        start_time = time.time()
        accum_loss, accum_count = 0, 0

        for batch in L:
            batch = batch.to(device)
            
            protein_residue = batch.protein_residue
            protein_pos_N = batch.protein_pos[batch.protein_mask_n]
            protein_pos_Ca = batch.protein_pos[batch.protein_mask_ca]
            protein_pos_C = batch.protein_pos[batch.protein_mask_c]
            
            peptide_residue = batch.peptide_residue
            peptide_pos_N = batch.peptide_pos[batch.peptide_mask_n]
            peptide_pos_Ca = batch.peptide_pos[batch.peptide_mask_ca]
            peptide_pos_C = batch.peptide_pos[batch.peptide_mask_c]

            y = batch.energy
            
            y_pred = binding_model(
                protein_residue, protein_pos_N, protein_pos_Ca, protein_pos_C, batch.protein_batch,
                peptide_residue, peptide_pos_N, peptide_pos_Ca, peptide_pos_C, batch.peptide_batch,
            )

            loss = criterion(y, y_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accum_loss += loss.item()
            accum_count += 1
            
        print("epoch: {}\tloss pos: {:.5f}\t{:.3f}s".format(e, accum_loss / accum_count, time.time() - start_time))
    
    save_model(save_best=False)