import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        embed_dim: Dimension of token embeddings
        num_heads: Number of attention heads
        dropout: Dropout rate for attention weights
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Dimension per head
        self.scale = self.head_dim ** -0.5  # Scaling factor for dot product attention

        # Linear layers for Q, K, V transformations
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, num_patches, embed_dim]
        return: Attention output of shape [batch_size, num_patches, embed_dim]
        """
        batch_size, num_patches, embed_dim = x.shape

        # Compute Q, K, V (split into num_heads)
        qkv = self.qkv(x)  # Shape: [batch_size, num_patches, 3 * embed_dim]
        qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: [3, batch_size, num_heads, num_patches, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Split into Q, K, V

        # Compute scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, num_patches, num_patches]
        attn_weights = attn_scores.softmax(dim=-1)  # Apply softmax
        attn_weights = self.attn_drop(attn_weights)  # Apply dropout

        # Compute attention output
        attn_output = attn_weights @ v  # [batch_size, num_heads, num_patches, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, num_patches, embed_dim)  # Merge heads

        # Apply final projection
        output = self.proj(attn_output)
        output = self.proj_drop(output)
        return output
