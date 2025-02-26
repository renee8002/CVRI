import torch
from model.attention import MultiHeadSelfAttention

# Create Multi-Head Self-Attention instance
attn = MultiHeadSelfAttention(embed_dim=768, num_heads=12)

# Generate a dummy input (batch=2, num_patches=196, embed_dim=768)
dummy_input = torch.randn(2, 196, 768)

# Forward pass through attention module
output = attn(dummy_input)

# Print output shape
print(f"Output shape: {output.shape}")  # Expected: [2, 196, 768]
