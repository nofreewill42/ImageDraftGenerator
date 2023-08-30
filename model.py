import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, num_colors=216, d_model=512, nhead=8, num_layers=12, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=True):#, return_intermediate_dec=False):
        super().__init__()
        self.embeddings = nn.Embedding(num_colors, d_model)
        L = 4096
        self.positional_encoding = nn.Parameter(torch.randn(1, L, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.palette_predictor = nn.Linear(d_model, num_colors)

    def forward(self, x):
        x = self.embeddings(x)
        x = x + self.positional_encoding[:, :x.shape[1]]
        mask = self.generate_square_subsequent_mask(x.shape[1], device=x.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        x = self.palette_predictor(x)
        return x
    
    
    @staticmethod
    def generate_square_subsequent_mask(sz: int, device='cpu') -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)