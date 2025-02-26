import torch
from model.patch_embed import PatchEmbedding

# 创建 Patch Embedding 实例
patch_embed = PatchEmbedding(image_size=224, patch_size=16, embed_dim=768)

# 生成一个 batch 的随机图片 (batch=2, channels=3, height=224, width=224)
dummy_img = torch.randn(2, 3, 224, 224)

# 通过 Patch Embedding
output = patch_embed(dummy_img)

# 打印输出形状
print(f"Output shape: {output.shape}")  # 预期: [2, 196, 768]
