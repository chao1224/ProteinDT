import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import time

import torch
from transformers import BertTokenizer, T5Tokenizer
from torch.utils.data import DataLoader

from ProteinDT.datasets import RepresentationPairWithRawDataDataset
# from ProteinDT.models import GaussianSDEDecoderModel, ColdDiffusionDecoder, LatentDiffusionDecoder, MultinomialDiffusion, LSTMDecoder, T5Decoder
from ProteinDT.models import MultinomialDiffusion, T5Decoder


def eval(dataloader):
    decoder_distribution_model.eval()
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader

    fuzzy_match_acc_accum, exact_match_acc_accum = 0, 0
    for batch_idx, batch in enumerate(L):
        protein_sequence = batch["protein_sequence"]
            
        protein_sequence_encode = protein_decoder_tokenizer(protein_sequence, truncation=True, max_length=args.protein_max_sequence_len, padding='max_length', return_tensors='pt')

        protein_sequence_input_ids = protein_sequence_encode.input_ids.squeeze().to(device)  # (B, max_seq_len)
        protein_sequence_attention_mask = protein_sequence_encode.attention_mask.squeeze().to(device)  # (B, max_seq_len)

        raw_input_sequence_list = protein_decoder_tokenizer.batch_decode(sequences=protein_sequence_input_ids, skip_special_tokens=True)
        for i in range(5):
            raw_input_ids = protein_sequence_input_ids[i].cpu().numpy()
            raw_input_sequence = raw_input_sequence_list[i].replace(" ", "")
            print("id:", i, len(raw_input_ids), len(raw_input_sequence))
            print(raw_input_ids[:20], raw_input_ids[-20:])
            print(raw_input_sequence[:20], raw_input_sequence[-20:])
            print()
        print()

        protein_repr = batch["protein_repr"].to(device)
        if args.decoder_distribution in ["T5Decoder"]:
            protein_sequence_pred_ids = decoder_distribution_model.inference(condition=protein_repr, protein_seq_attention_mask=protein_sequence_attention_mask, max_seq_len=args.protein_max_sequence_len)
        elif args.decoder_distribution in ["LSTMDecoder"]:
            protein_sequence_pred_ids = decoder_distribution_model.inference(condition=protein_repr, protein_seq_attention_mask=protein_sequence_attention_mask, max_seq_len=args.protein_max_sequence_len)
        else:
            protein_sequence_output = decoder_distribution_model.inference(condition=protein_repr, protein_seq_attention_mask=protein_sequence_attention_mask, max_seq_len=args.protein_max_sequence_len)
            protein_sequence_pred_ids = torch.argmax(protein_sequence_output, dim=-1)  # (B, max_seq_len)

        pred_sequence_list = protein_decoder_tokenizer.batch_decode(sequences=protein_sequence_pred_ids, skip_special_tokens=True)
        for i in range(5):
            print("ground-truth", protein_sequence_input_ids[i])
            print("prediction", protein_sequence_pred_ids[i])
            pred_sequence = pred_sequence_list[i].replace(" ", "")
            print("pred", i, len(pred_sequence))
            print("prediction", pred_sequence[:20], pred_sequence[-20:])
            print()
        print()

        if args.decoder_distribution in ["T5Decoder"]:
            fuzzy_match_count = (protein_sequence_input_ids[:, :-1] == protein_sequence_pred_ids[:, 1:])  # (B, max_seq_len)
            exact_match_count = fuzzy_match_count * protein_sequence_attention_mask[:, :-1]  # (B, max_seq_len)
        else:
            fuzzy_match_count = (protein_sequence_input_ids == protein_sequence_pred_ids)  # (B, max_seq_len)
            exact_match_count = fuzzy_match_count * protein_sequence_attention_mask  # (B, max_seq_len)

        B, max_len = fuzzy_match_count.size()
        fuzzy_match_acc = 100. * fuzzy_match_count.sum().item() / (B * max_len)
        exact_match_acc = 100. * exact_match_count.sum().item() / protein_sequence_attention_mask.sum().item()
        fuzzy_match_acc_accum += fuzzy_match_acc
        exact_match_acc_accum += exact_match_acc

        print("fuzzy: {:.2f}\texact: {:.2f}".format(fuzzy_match_acc, exact_match_acc))
        exit()
    
    # fuzzy_match_acc_accum /= len(L)
    # exact_match_acc_accum /= len(L)
    # print("fuzzy match acc: {:.5f}".format(fuzzy_match_acc_accum))
    # print("exact match acc: {:.5f}".format(exact_match_acc_accum))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--condition_dim", type=int, default=256)
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)

    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)

    parser.add_argument("--decoder_distribution", type=str, default="GaussianSDE", choices=["GaussianSDE", "LSTMDecoder", "T5Decoder", "ColdDiffusion", "LatentDiffusion", "MultinomialDiffusion"])
    parser.add_argument("--pretrained_folder", type=str, default=None)
    parser.add_argument("--step_04_folder", type=str, default=None)

    # for LSTM
    parser.add_argument("--LSTM_layer", type=int, default=5)
    parser.add_argument("--LSTM_epsilon", type=float, default=0.05)

    parser.add_argument("--beta_min", type=float, default=0.1)
    parser.add_argument("--beta_max", type=float, default=30)
    parser.add_argument("--num_diffusion_timesteps", type=int, default=1000)
    parser.add_argument("--SDE_type", type=str, default="VP")
    parser.add_argument("--score_network_type", type=str, default="BertProtBFD")

    args = parser.parse_args()
    print("arguments", args)
    assert args.pretrained_folder is not None
    assert args.step_04_folder is not None

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    if args.decoder_distribution in ["T5Decoder"]:
        protein_decoder_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False, chache_dir="../data/temp_pretrained_t5_base")
    else:
        protein_decoder_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, chache_dir="../data/temp_pretrained_ProtBert")

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
            T5_model=args.score_network_type
        )
    decoder_distribution_model.to(device)

    ########## Load model weight ##########
    step_04_folder = args.step_04_folder
    model_file_path = os.path.join(step_04_folder, "decoder_distribution_model.pth")
    print("Loading decoder model from {}...".format(model_file_path))
    state_dict = torch.load(model_file_path, map_location='cpu')
    decoder_distribution_model.load_state_dict(state_dict)
    decoder_distribution_model.eval()

    step_02_folder = os.path.join(args.pretrained_folder, "step_02_pairwise_representation")
    dataset = RepresentationPairWithRawDataDataset(step_02_folder, prob_unconditional=0)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    eval(dataloader)
    """
        0  [PAD]
        1  [UNK]
        2  [CLS]
        3  [SEP]
        4  [MASK]
        5  L
        6  A
        7  G
        8  V
        9  E
        10 S
        11 I
        12 K
        13 R
        14 D
        15 T
        16 P
        17 N
        18 Q
        19 F
        20 Y
        21 M
        22 H
        23 C
        24 W
        25 X
        26 U
        27 B
        28 Z
        29 O
    """
