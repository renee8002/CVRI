import torch
from model.transformer import TransformerEncoder

# Create Transformer Encoder instance
encoder = TransformerEncoder(embed_dim=768, num_heads=12)

# Generate a dummy input (batch=2, num_patches=196, embed_dim=768)
dummy_input = torch.randn(2, 196, 768)

# Forward pass through Transformer Encoder
output = encoder(dummy_input)

# Print output shape
print(f"Output shape: {output.shape}")  # Expected: [2, 196, 768]
