import torch
from model.mlp import MLPBlock

# Create MLP Block instance
mlp = MLPBlock(embed_dim=768, hidden_dim=3072)

# Generate a dummy input (batch=2, num_patches=196, embed_dim=768)
dummy_input = torch.randn(2, 196, 768)

# Forward pass through MLP block
output = mlp(dummy_input)

# Print output shape
print(f"Output shape: {output.shape}")  # Expected: [2, 196, 768]
