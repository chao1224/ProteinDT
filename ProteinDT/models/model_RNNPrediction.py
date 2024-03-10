import torch
from torch import nn


class RNNPrediction(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.rnn_layer = nn.RNN(self.hidden_dim, self.hidden_dim, self.num_layers)
        self.mlp_layer = nn.Sequential(
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        return
    
    def forward(self, protein_seq_repr):
        max_seq_len = protein_seq_repr.size()[1]

        device = protein_seq_repr.device
        h0 = torch.randn(self.num_layers, max_seq_len, self.hidden_dim).to(device=device)  # (num_layers, max_seq_len, hidden_dim)

        out, _ = self.rnn_layer(protein_seq_repr, h0)  # (B, max_seq_len, hidden_dim), (num_layers, max_seq_len, hidden_dim)
        out = self.mlp_layer(out)  # (B, max_seq_len, 1)
        return out
