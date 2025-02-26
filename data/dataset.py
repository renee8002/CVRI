# data/dataset.py

import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CIFAR10Dataset:
    def __init__(self, root_dir="./data", train=True, image_size=224):
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            # transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Add augmentation for training
        if train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        # Load CIFAR-10 dataset
        self.dataset = datasets.CIFAR10(
            root=root_dir,
            train=train,
            download=True,
            transform=self.transform
        )

    def get_dataloader(self, batch_size=64, shuffle=True, num_workers=4):
        if num_workers is None:
            num_workers = 0 if os.name == "nt" or "darwin" in os.sys.platform else 4

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )


# Usage example:
def get_dataloaders(batch_size=64, image_size=224, num_workers=0):
    train_dataset = CIFAR10Dataset(train=True, image_size=image_size)
    val_dataset = CIFAR10Dataset(train=False, image_size=image_size)

    train_loader = train_dataset.get_dataloader(batch_size=batch_size)
    val_loader = val_dataset.get_dataloader(batch_size=batch_size, shuffle=False)

    return train_loader, val_loader