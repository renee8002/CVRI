import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        """
        embed_dim: Input dimension (same as embedding dimension in ViT)
        hidden_dim: Hidden layer dimension (usually 4x embed_dim)
        dropout: Dropout rate for regularization
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)  # First linear layer
        self.act = nn.GELU()  # Activation function
        self.fc2 = nn.Linear(hidden_dim, embed_dim)  # Second linear layer
        self.dropout = nn.Dropout(dropout)  # Dropout layer

    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, num_patches, embed_dim]
        return: Output tensor of the same shape [batch_size, num_patches, embed_dim]
        """
        x = self.fc1(x)  # Apply first linear layer
        x = self.act(x)  # Apply activation function
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Apply second linear layer
        x = self.dropout(x)  # Apply dropout again
        return x
