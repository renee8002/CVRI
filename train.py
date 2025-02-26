# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from model.vit import ViT
from data.dataset import get_dataloaders
from utils.training import accuracy, save_model
from utils.visualization import plot_training_curves

def train():
    # Hyperparameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 3e-4  # ViT typically uses a higher learning rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size,
        image_size=224,
        num_workers=0
    )

    # Initialize ViT model
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=10
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Lists to store training and validation metrics
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # Training loop
    for epoch in range(num_epochs):
        print("Training started...")
        model.train()
        total_loss, total_correct = 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                acc = total_correct / ((batch_idx + 1) * batch_size)
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")

        # Compute training accuracy and loss
        train_acc = total_correct / len(train_loader.dataset)
        train_losses.append(total_loss / len(train_loader))
        train_accs.append(train_acc)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f}")

        # Validation loop
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        # Compute validation accuracy and loss
        val_acc = val_correct / len(val_loader.dataset)
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step()

    # Save the trained model
    save_model(model, "vit_cifar10.pth")

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

if __name__ == "__main__":
    train()
