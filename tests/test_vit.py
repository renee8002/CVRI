import torch
from model.vit import ViT

# Create ViT instance
vit = ViT(image_size=224, patch_size=16, num_classes=10)

# Generate a dummy input (batch=2, 3 channels, 224x224 image)
dummy_input = torch.randn(2, 3, 224, 224)

# Forward pass through ViT
output = vit(dummy_input)

# Print output shape
print(f"Output shape: {output.shape}")  # Expected: [2, 10]
