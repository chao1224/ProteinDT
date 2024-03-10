# This script is merged into the dataset scripts. I.e., it is no longer used, only for backup.

import re
import numpy as np
import torch


# This is for ProtBERT
def preprocess_each_protein_sequence(sentence, tokenizer, max_seq_len):
    sentence = re.sub(r"[UZOB]", "X", sentence)
    sentence = " ".join(sentence)
    text_input = tokenizer(sentence, truncation=True, max_length=max_seq_len, padding='max_length', return_tensors='np')
    
    input_ids = text_input['input_ids'].squeeze()
    attention_mask = text_input['attention_mask'].squeeze()
    return [input_ids, attention_mask]


# This is for ProtBERT
def prepare_protein_sequence_tokens(description, tokenizer, max_seq_len):
    B = len(description)
    tokens_outputs = [preprocess_each_protein_sequence(description[idx], tokenizer, max_seq_len) for idx in range(B)]
    tokens_ids = [o[0] for o in tokens_outputs]
    masks = [o[1] for o in tokens_outputs]
    tokens_ids = torch.Tensor(tokens_ids).long()
    masks = torch.Tensor(masks).bool()
    return tokens_ids, masks


# This is for SciBERT
def preprocess_each_text_sentence(sentence, tokenizer, max_seq_len):
    text_input = tokenizer(sentence, truncation=True, max_length=max_seq_len, padding='max_length', return_tensors='np')
    
    input_ids = text_input['input_ids'].squeeze()
    attention_mask = text_input['attention_mask'].squeeze()

    return [input_ids, attention_mask]


# This is for SciBERT
def prepare_text_sequence_tokens(description, tokenizer, max_seq_len):
    B = len(description)
    tokens_outputs = [preprocess_each_text_sentence(description[idx], tokenizer, max_seq_len) for idx in range(B)]
    tokens_ids = [o[0] for o in tokens_outputs]
    masks = [o[1] for o in tokens_outputs]
    tokens_ids = torch.Tensor(tokens_ids).long()
    masks = torch.Tensor(masks).bool()
    return tokens_ids, masks