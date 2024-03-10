import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# from utils import *


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim, n_layer, embedding_dim, epsilon, num_classes, tokenizer):
        super(LSTMDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.epsilon = epsilon

        self.embedding_layer = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.embedding_dim)

        self.lstm = nn.LSTM(input_size=self.hidden_dim+self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.n_layer, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)
        self.CE_criterion = nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        return
        
    def forward(self, protein_seq_input_ids, protein_seq_attention_mask, condition):
        input_seq_emb = self.embedding_layer(protein_seq_input_ids)  # [B, max_seq_len, embedding_dim]

        batch_size, n_seq, _ = input_seq_emb.size()

        h_0 = torch.zeros(self.n_layer, batch_size, self.hidden_dim).detach()
        c_0 = torch.zeros(self.n_layer, batch_size, self.hidden_dim).detach() 
        g_hidden = (h_0.cuda(), c_0.cuda())

        condition = condition.unsqueeze(1) # [B, 1, hidde_dim]
        prev_word = protein_seq_input_ids[:, 0:1]  # [B, 1]

        output = []

        for j in range(n_seq-1):
            if random.random() < self.epsilon:
                current_word_emb = self.embedding_layer(prev_word)  # [B, 1, embedding_dim]
            else:
                current_word_emb = input_seq_emb[:, j:(j+1), :]  # [B, 1, embedding_dim]

            x = torch.cat([condition, current_word_emb], dim=-1)  # [B, 1, hidden_dim+embedding_dim]
            logits, g_hidden = self.lstm(x, g_hidden)  # [B, 1, hidden_dim], [[n_layer, B, hidden_dim], [n_layer, B, hidden_dim]]
            logits = self.fc(logits)  # [B, 1, num_classes]
            prev_word = torch.argmax(logits, dim = -1)  # [B, 1]
            output.append(logits)

        output = torch.cat(output, dim=1)  # [B, max_seq_len-1, num_classes]
        target_protein_seq_input_ids = protein_seq_input_ids[:, 1:].contiguous()  # [B, max_seq_len-1]
        target_protein_seq_attention_mask = protein_seq_attention_mask[:, 1:].contiguous()  # [B, max_seq_len-1]
        flattened_logits = output.view(-1, output.size(-1))  # [B * (max_sequence_len-1), num_class]
        flattened_ids = target_protein_seq_input_ids.view(-1)  # [B * (max_sequence_len-1)]
        flattened_mask = target_protein_seq_attention_mask.view(-1)  # [B * (max_sequence_len-1)]
        total_loss = self.CE_criterion(flattened_logits, flattened_ids)  # [B * (max_sequence_len-1)]
        masked_loss = total_loss * flattened_mask  # [B * (max_sequence_len-1)]
        total_loss = torch.mean(total_loss)
        masked_loss = masked_loss.sum() / flattened_mask.sum()

        loss = total_loss + masked_loss
        decoding_loss = 0

        return loss, decoding_loss

    def inference(self, condition, protein_seq_attention_mask, max_seq_len, temperature=1, use_sample=False):

        device = condition.device
        condition = condition.unsqueeze(1) # [B, 1, hidde_dim]
        batch_size = condition.size()[0]
        prev_word = torch.ones([batch_size]).long().to(device) * self.tokenizer.cls_token_id # [B]
        prev_word = prev_word.unsqueeze(1) # [B, 1]

        h_0 = torch.zeros(self.n_layer, batch_size, self.hidden_dim).detach()
        c_0 = torch.zeros(self.n_layer, batch_size, self.hidden_dim).detach() 
        g_hidden = (h_0.cuda(), c_0.cuda())

        output = []
        for _ in range(max_seq_len):
            current_word_emb = self.embedding_layer(prev_word)  # [B, 1, embedding_dim]
            x = torch.cat([condition, current_word_emb], dim=-1)  # [B, 1, hidden_dim+embedding_dim]
            logits, g_hidden = self.lstm(x, g_hidden)  # [B, 1, hidden_dim], [[n_layer, B, hidden_dim], [n_layer, B, hidden_dim]]
            logits = self.fc(logits)  # [B, 1, num_classes]

            if use_sample:
                probs = torch.softmax(logits / temperature, dim=-1) 
                prediction = []
                for data_idx in range(batch_size):
                    prediction_temp = torch.multinomial(probs[data_idx], num_samples=1)
                    prediction.append(prediction_temp)
                prediction = torch.cat(prediction) # [B, 1]
                prev_word = prediction  # [B, 1]
            else:
                prev_word = torch.argmax(logits, dim=-1)  # [B, 1]

            output.append(prev_word)

        output = torch.cat(output, dim=1)  # [B, max_seq_len]
        return output
