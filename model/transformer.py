import torch
import torch.nn as nn

from model.attention import AttentionPooling, MultiHeadSelfAttention
from model.feed_forward import PositionWiseFeedForward
from model.positional_encoding import PositionalEncoding


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-LN Transformer (more stable training)
        normed = self.norm1(x)
        x = x + self.dropout1(self.attn(normed, mask))
        normed = self.norm2(x)
        x = x + self.dropout2(self.ffn(normed))
        return x


class StockTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        pred_len: int = 5,
        dropout: float = 0.1,
        seq_len: int = 60,
        pos_encoding: str = "sinusoidal",
        pooling: str = "attention",
    ):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len

        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoding = PositionalEncoding(
            d_model, max_len=seq_len, learned=(pos_encoding == "learned")
        )
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        if pooling == "attention":
            self.pool = AttentionPooling(d_model)
        else:
            self.pool = None  # mean pooling

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_len),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        # x: (batch, seq_len, num_features)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)  # (batch, seq_len, d_model)

        for block in self.encoder_blocks:
            x = block(x, mask)  # (batch, seq_len, d_model)

        x = self.norm(x)

        if self.pool is not None:
            pooled = self.pool(x)  # (batch, d_model)
        else:
            pooled = x.mean(dim=1)  # (batch, d_model)

        out = self.output_head(pooled)  # (batch, pred_len)
        return out

    def get_attention_weights(self) -> list:
        weights = []
        for block in self.encoder_blocks:
            if block.attn.attn_weights is not None:
                weights.append(block.attn.attn_weights)
        return weights
