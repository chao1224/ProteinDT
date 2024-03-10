import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer

from utils import text_prompt_dict, load_oracle_evaluator, load_editing_dataset_and_loader


@torch.no_grad()
def select(dataloader):
    eval_prediction_model.eval()

    protein_sequence_list = []
    logits_list = []
    target_list = []
    eval_list = []
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    for batch_idx, batch in enumerate(L):
        protein_sequence = batch["protein_sequence"]
        protein_sequence_input_ids = batch["protein_sequence_input_ids"].to(device)
        protein_sequence_attention_mask = batch["protein_sequence_attention_mask"].to(device)
        target = batch["label"].to(device)
        
        output = eval_prediction_model(protein_sequence_input_ids, protein_sequence_attention_mask)
        logits = output.logits

        if args.editing_task in ["alpha", "beta"]:
            pred = logits.argmax(dim=-1, keepdim=False)
            pred = torch.where(protein_sequence_attention_mask==1, pred, -1)
            eval_result = pred.eq(target)
            
            total = protein_sequence_attention_mask.sum(-1)
            eval_result = eval_result * protein_sequence_attention_mask
            eval_result = eval_result.sum(-1)
            logits = eval_result # pseudo logits, without any specific meanings
            target = eval_result
        else:
            logits = logits.squeeze(1)
            eval_result = criterion_eval(logits, target)
        
        protein_sequence_list.extend(protein_sequence)
        logits_list.append(logits.detach().cpu().numpy())
        target_list.append(target.detach().cpu().numpy())
        eval_list.append(eval_result.detach().cpu().numpy())

    logits_list = np.concatenate(logits_list)
    target_list = np.concatenate(target_list)
    eval_list = np.concatenate(eval_list)
    print("len of protein_sequence_list", len(protein_sequence_list))
    print("logits_list: {}".format(logits_list.shape))
    print("target_list: {}".format(target_list.shape))
    print("eval_list: {}".format(eval_list.shape))
    
    output_path = os.path.join(output_folder, "preprocessed_data.csv")
    f = open(output_path, 'w')
    for protein_sequence, logits, target, eval in zip(protein_sequence_list, logits_list, target_list, eval_list):
        protein_sequence = protein_sequence.replace(" ", "")
        print("{},{},{},{}".format(protein_sequence, target, logits, eval), file=f)
        
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dataset_size", type=int, default=200)
    parser.add_argument("--output_model_dir", type=str, default=None)

    parser.add_argument("--editing_task", type=str, default="Villin")

    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=True)

    parser.add_argument("--protein_backbone_model", type=str, default="ProtBERT_BFD", choices=["ProtBERT", "ProtBERT_BFD"])
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)
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
    
    ##### Create prediction head
    eval_prediction_model, eval_protein_tokenizer = load_oracle_evaluator(args.editing_task, device)
    test_dataset, test_dataloader, criterion_eval = load_editing_dataset_and_loader(args, eval_protein_tokenizer)
    print("len of test: {}".format(len(test_dataset)))

    output_folder = text_prompt_dict[args.editing_task]["data_folder"]
    select(test_dataloader)
