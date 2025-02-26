import torch
from utils.training import accuracy

# Generate dummy output (logits) and labels
outputs = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.9, 0.05, 0.05]])
labels = torch.tensor([2, 1, 0])

# Compute accuracy
acc = accuracy(outputs, labels)
print(f"Accuracy: {acc:.2f}")  # Expected: 1.00 (100% accuracy)
