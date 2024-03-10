import torch.nn as nn


class GaussianFacilitatorModel(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.MLP = nn.Sequential(
            nn.Linear( self.latent_dim, self.latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.criterion = nn.MSELoss()
        return
    
    def forward(self, protein_repr, text_repr):
        protein_repr_pred = self.MLP(text_repr)
        loss = self.criterion(protein_repr, protein_repr_pred)
        return loss
    
    def inerence(self, text_repr):
        protein_repr_pred = self.MLP(text_repr)
        return protein_repr_pred