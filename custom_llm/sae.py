import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_lambda=1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_lambda = sparsity_lambda

    def forward(self, x, punir_ids=None, reforcar_ids=None, punir_peso=1e-2, reforcar_peso=1e-2, return_hidden=False):
        z = torch.sigmoid(self.encoder(x))
        x_recon = self.decoder(z)
        
        sparsity_loss = self.sparsity_lambda * torch.abs(z).mean()

        extra_loss = 0

        if punir_ids is not None:
            extra_loss += punir_peso * z[:, punir_ids].abs().mean()

        if reforcar_ids is not None:
            ativacao = z[:, reforcar_ids]
            extra_loss += reforcar_peso * ((1 - ativacao).abs().mean())

        total_loss = sparsity_loss + extra_loss

        if return_hidden:
            return x_recon, total_loss, z
        else:
            return x_recon, total_loss
