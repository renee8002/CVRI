import torch
import torch.nn.functional as F

def accuracy(outputs, labels):
    """
    Compute accuracy of the model predictions.
    outputs: Tensor of shape [batch_size, num_classes]
    labels: Tensor of shape [batch_size]
    return: Accuracy as a float value
    """
    preds = outputs.argmax(dim=1)  # Get class with highest probability
    correct = preds.eq(labels).sum().item()
    return correct / labels.size(0)

def save_model(model, path="vit_cifar10.pth"):
    """
    Save model state dictionary to a file.
    model: The model instance to be saved.
    path: File path for saving the model.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="vit_cifar10.pth", device="cpu"):
    """
    Load model state dictionary from a file.
    model: The model instance to load weights into.
    path: File path of saved model.
    device: Device to load the model onto ("cpu" or "cuda").
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")
    return model