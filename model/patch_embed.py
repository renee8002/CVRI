
#
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"

        # 计算 patch 的数量
        self.num_patches = (image_size // patch_size) ** 2

        # Conv2d 实现 Patch Embedding，相当于把每个 patch 视为一个 token
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 位置编码，告诉 Transformer 每个 Patch 在图片中的位置
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x):
        """
        x: 输入图片张量，形状 [batch_size, 3, image_size, image_size]
        return: 形状 [batch_size, num_patches, embed_dim] 的序列
        """
        x = self.proj(x)  # [batch, embed_dim, num_patches_height, num_patches_width]
        x = x.flatten(2)  # 展平 num_patches 维度，变成 [batch, embed_dim, num_patches]
        x = x.transpose(1, 2)  # 交换维度，变成 [batch, num_patches, embed_dim]
        x = x + self.pos_embed  # 加上位置编码
        return x
