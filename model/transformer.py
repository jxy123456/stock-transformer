"""StockMultiHorizonTransformer: Transformer Encoder + 双周期分类头。"""

import torch
import torch.nn as nn

from model.base import BaseModel


class StockMultiHorizonTransformer(BaseModel):
    def __init__(
        self,
        feature_dim=39,
        seq_len=120,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.20,
        num_bins_5d=11,
        num_bins_20d=11,
    ):
        super().__init__()

        self.feature_norm = nn.LayerNorm(feature_dim)
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                             enable_nested_tensor=False)

        self.head_5d = self._make_head(d_model, 64, num_bins_5d, dropout)
        self.head_20d = self._make_head(d_model, 64, num_bins_20d, dropout)

        self._init_weights()

    def _make_head(self, d_model, hidden_dim, out_dim, dropout):
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, padding_mask=None):
        x = self.feature_norm(x)
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :x.size(1), :]

        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        last_hidden = encoded[:, -1, :]

        return {
            "logits_5d": self.head_5d(last_hidden),
            "logits_20d": self.head_20d(last_hidden),
        }
