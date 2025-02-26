import torch
import torch.nn as nn
from model.patch_embed import PatchEmbedding
from model.transformer import TransformerEncoder

class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, num_classes=10,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        """
        image_size: Input image size (e.g., 224 for ImageNet)
        patch_size: Patch size (e.g., 16x16)
        in_channels: Number of image channels (e.g., 3 for RGB)
        num_classes: Number of output classes (e.g., 10 for CIFAR-10)
        embed_dim: Embedding dimension
        depth: Number of Transformer Encoder layers
        num_heads: Number of attention heads
        mlp_ratio: Expansion ratio for MLP
        dropout: Dropout rate
        """
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable class token ([CLS])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Position Embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        # Transformer Encoder stack
        self.encoders = nn.Sequential(*[
            TransformerEncoder(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        # Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: Input image tensor of shape [batch_size, 3, image_size, image_size]
        return: Output logits of shape [batch_size, num_classes]
        """
        batch_size = x.shape[0]

        # Patch Embedding
        x = self.patch_embed(x)  # Shape: [batch_size, num_patches, embed_dim]

        # Add class token
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, embed_dim]
        x = torch.cat([cls_token, x], dim=1)  # [batch_size, num_patches+1, embed_dim]

        # Add position embedding
        x = x + self.pos_embed  # [batch_size, num_patches+1, embed_dim]

        # Forward through Transformer Encoder layers
        x = self.encoders(x)  # [batch_size, num_patches+1, embed_dim]

        # Take only [CLS] token output
        x = self.norm(x[:, 0])  # [batch_size, embed_dim]

        # Classification Head
        x = self.head(x)  # [batch_size, num_classes]

        return x
