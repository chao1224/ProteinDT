import torch
import torch.nn as nn


class RNNScoreNetwork(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.rnn_layer = nn.RNN(self.hidden_dim, self.hidden_dim, self.num_layers)
        self.mlp_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        return
    
    def forward(self, protein_seq_repr, protein_seq_attention_mask, condition):
        """
        Args:
            protein_seq_repr: noised protein token-level representation, (B, max_seq_len, hidden_dim)
            protein_seq_attention_mask: masking, (B, max_seq_len)
            condition: the condition matrix, (B, hidden_dim)
        Output:
            gradient (score)
        """
        max_seq_len = protein_seq_repr.size()[1]
        protein_seq_attention_mask = protein_seq_attention_mask.unsqueeze(2)  # (B, max_seq_len, 1)
        condition = condition.unsqueeze(1).expand(-1, max_seq_len, -1)  # (B, max_seq_len, hidden_dim)
        condition = condition * protein_seq_attention_mask  # (B, max_seq_len, hidden_dim)

        conditioned_protein_seq_repr = protein_seq_repr + condition  # (B, max_seq_len, hidden_dim)

        max_seq_len = conditioned_protein_seq_repr.size()[1]
        device = protein_seq_repr.device
        h0 = torch.randn(self.num_layers, max_seq_len, self.hidden_dim).to(device=device)  # (num_layers, max_seq_len, hidden_dim)

        conditioned_protein_seq_repr, _ = self.rnn_layer(conditioned_protein_seq_repr, h0)  # (B, max_seq_len, hidden_dim), (num_layers, max_seq_len, hidden_dim)
        score = self.mlp_layer(conditioned_protein_seq_repr)  # (B, max_seq_len, 1)
        return score