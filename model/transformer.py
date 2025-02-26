import torch
import torch.nn as nn
from model.attention import MultiHeadSelfAttention
from model.mlp import MLPBlock


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        """
        embed_dim: Embedding dimension (same as token embedding)
        num_heads: Number of attention heads
        mlp_ratio: Expansion ratio for MLP (typically 4x embed_dim)
        dropout: Dropout rate for regularization
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)  # LayerNorm before Attention
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)  # LayerNorm before MLP
        self.mlp = MLPBlock(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, num_patches, embed_dim]
        return: Output tensor of the same shape [batch_size, num_patches, embed_dim]
        """
        # Apply LayerNorm -> Self-Attention -> Residual Connection
        x = x + self.attn(self.norm1(x))

        # Apply LayerNorm -> MLP Block -> Residual Connection
        x = x + self.mlp(self.norm2(x))

        return x
