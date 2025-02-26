import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """
    Plot training & validation loss and accuracy curves.
    train_losses: List of training losses per epoch
    val_losses: List of validation losses per epoch
    train_accs: List of training accuracies per epoch
    val_accs: List of validation accuracies per epoch
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.plot(epochs, val_losses, "r-", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # Plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, "b-", label="Training Accuracy")
    plt.plot(epochs, val_accs, "r-", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    plt.show()


def visualize_attention(attention_map, image, patch_size=16):
    """
    Visualize the attention map on an input image.
    attention_map: Tensor of shape [num_heads, num_patches, num_patches]
    image: Original input image (should be in PIL format or numpy array)
    patch_size: Size of each patch (e.g., 16 for ViT)
    """
    attention_map = attention_map.mean(dim=0).detach().cpu().numpy()  # Average over heads
    num_patches = attention_map.shape[0]
    grid_size = int(np.sqrt(num_patches))

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.imshow(attention_map.reshape(grid_size, grid_size), cmap="jet", alpha=0.6)
    plt.title("Attention Map")
    plt.axis("off")
    plt.show()
