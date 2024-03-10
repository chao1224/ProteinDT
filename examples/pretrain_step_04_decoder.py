import os
import random
import numpy as np
import argparse
from tqdm import tqdm
import time

import torch
from torch import nn
import torch.optim as optim

from transformers import BertTokenizer, T5Tokenizer
from torch.utils.data import DataLoader


from ProteinDT.datasets import RepresentationPairWithRawDataDataset
# from ProteinDT.models import GaussianSDEDecoderModel, ColdDiffusionDecoder, LatentDiffusionDecoder, MultinomialDiffusion, LSTMDecoder, T5Decoder
from ProteinDT.models import MultinomialDiffusion, T5Decoder


def save_model(save_best):
    if save_best:
        global optimal_loss
        print("save model with loss: {:.5f}".format(optimal_loss))
        model_file = "model.pth"
        
        saved_file_path = os.path.join(step_04_folder, "decoder_distribution_{}".format(model_file))
        torch.save(decoder_distribution_model.state_dict(), saved_file_path)

    else:
        model_file = "model_final.pth"
        
        saved_file_path = os.path.join(step_04_folder, "decoder_distribution_{}".format(model_file))
        torch.save(decoder_distribution_model.state_dict(), saved_file_path)
    return


def train(dataloader):
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader

    start_time = time.time()
    accum_SDE_loss, accum_decoding_loss = 0, 0
    for batch_idx, batch in enumerate(L):
        protein_sequence = batch["protein_sequence"]
            
        protein_sequence_encode = protein_decoder_tokenizer(protein_sequence, truncation=True, max_length=args.protein_max_sequence_len, padding='max_length', return_tensors='pt')
        protein_sequence_input_ids = protein_sequence_encode.input_ids.squeeze(1).to(device)
        protein_sequence_attention_mask = protein_sequence_encode.attention_mask.squeeze(1).to(device)

        protein_repr = batch["protein_repr"].to(device)

        SDE_loss, decoding_loss = decoder_distribution_model(condition=protein_repr, protein_seq_input_ids=protein_sequence_input_ids, protein_seq_attention_mask=protein_sequence_attention_mask)
        loss = args.alpha_1 * SDE_loss + args.alpha_2 * decoding_loss
        if args.verbose and batch_idx % 100 == 0:
            print("SDE Loss: {:.5f}\tDecoding Loss: {:.5f}".format(SDE_loss.item(), decoding_loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accum_SDE_loss += SDE_loss.item()
        accum_decoding_loss += decoding_loss.item()

    accum_SDE_loss /= len(L)
    accum_decoding_loss /= len(L)
    global optimal_loss
    temp_loss =  args.alpha_1 * accum_SDE_loss + args.alpha_2 * decoding_loss
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
    print("SDE Loss: {:.5f}\tDecoding Loss: {:.5f}\tTime: {:.5f}".format(accum_SDE_loss, accum_decoding_loss, time.time() - start_time))
    return


def train_AMP(dataloader):
    scaler = torch.cuda.amp.GradScaler()
    decoder_distribution_model.train()
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader

    start_time = time.time()
    accum_SDE_loss, accum_decoding_loss = 0, 0
    for batch_idx, batch in enumerate(L):
        protein_sequence = batch["protein_sequence"]
        
        protein_sequence_encode = protein_decoder_tokenizer(protein_sequence, truncation=True, max_length=args.protein_max_sequence_len, padding='max_length', return_tensors='pt')
        protein_sequence_input_ids = protein_sequence_encode.input_ids.squeeze(1).to(device)
        protein_sequence_attention_mask = protein_sequence_encode.attention_mask.squeeze(1).to(device)
        
        protein_repr = batch["protein_repr"].to(device)

        with torch.cuda.amp.autocast():
            SDE_loss, decoding_loss = decoder_distribution_model(protein_seq_input_ids=protein_sequence_input_ids, protein_seq_attention_mask=protein_sequence_attention_mask, condition=protein_repr)
            loss = args.alpha_1 * SDE_loss + args.alpha_2 * decoding_loss

        if args.verbose and batch_idx % 100 == 0:
            if torch.is_tensor(decoding_loss):
                print("SDE Loss: {:.5f}\tDecoding Loss: {:.5f}".format(SDE_loss.item(), decoding_loss.item()))
            else:
                print("SDE Loss: {:.5f}\tDecoding Loss: {:.5f}".format(SDE_loss.item(), decoding_loss))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        accum_SDE_loss += SDE_loss.item()
        if torch.is_tensor(decoding_loss):
            accum_decoding_loss += decoding_loss.item()

    accum_SDE_loss /= len(L)
    accum_decoding_loss /= len(L)
    global optimal_loss
    temp_loss =  args.alpha_1 * accum_SDE_loss + args.alpha_2 * decoding_loss
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
    print("SDE Loss: {:.5f}\tDecoding Loss: {:.5f}\tTime: {:.5f}".format(accum_SDE_loss, accum_decoding_loss, time.time() - start_time))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--condition_dim", type=int, default=256)
    parser.add_argument("--protein_backbone_model", type=str, default="ProtBert", choices=["ProtBert", "ProtBert_BFD"])
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay", type=float, default=0)

    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    
    parser.add_argument("--use_AMP", dest="use_AMP", action="store_true")
    parser.add_argument("--no_AMP", dest="use_AMP", action="store_false")
    parser.set_defaults(use_AMP=True)

    parser.add_argument("--decoder_distribution", type=str, default="T5Decoder", choices=["T5Decoder", "MultinomialDiffusion"])
    parser.add_argument("--pretrained_folder", type=str, default=None)
    parser.add_argument("--output_folder", type=str, default=None)

    # for GaussianSDE & diffusion
    parser.add_argument("--beta_min", type=float, default=0.1)
    parser.add_argument("--beta_max", type=float, default=30)
    parser.add_argument("--num_diffusion_timesteps", type=int, default=1000)
    parser.add_argument("--SDE_type", type=str, default="VP")
    parser.add_argument("--score_network_type", type=str, default="Toy")
    parser.add_argument("--alpha_1", type=float, default=1)
    parser.add_argument("--alpha_2", type=float, default=0)
    parser.add_argument("--prob_unconditional", type=float, default=0)

    args = parser.parse_args()
    print("arguments", args)
    assert args.pretrained_folder is not None
    assert args.output_folder is not None

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
        print(protein_decoder_tokenizer.get_vocab())
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
            T5_model=args.score_network_type)

    if torch.cuda.device_count() > 1:
        # parallel models
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(decoder_distribution_model)
        neo_batch_size = args.batch_size * torch.cuda.device_count()
        print("batch size from {} to {}".format(args.batch_size, neo_batch_size))
        args.batch_size = neo_batch_size
    decoder_distribution_model.to(device)

    model_param_group = [
        {"params": decoder_distribution_model.parameters(), "lr": args.lr},
    ]
    optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
    optimal_loss = 1e10

    step_02_folder = os.path.join(args.pretrained_folder, "step_02_pairwise_representation")
    dataset = RepresentationPairWithRawDataDataset(step_02_folder, prob_unconditional=args.prob_unconditional)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    step_04_folder = args.output_folder
    os.makedirs(step_04_folder, exist_ok=True)

    for e in range(1, 1+args.epochs):
        print("Epoch {}".format(e))
        if args.use_AMP:
            train_AMP(dataloader)
        else:
            train(dataloader)
