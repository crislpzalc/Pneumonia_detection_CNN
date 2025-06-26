"""
dataLoader.py
=============

Utility functions for loading the chest-X-ray dataset and creating
PyTorch DataLoaders for training, validation, and testing.

The directory structure is expected to be::

    <base_path>/
        train/
            NORMAL/
            PNEUMONIA/
        val/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/

Author
------
Cristina L. A., June 2025
"""

from typing import Tuple, List
import torch
import os
import torchvision
import torchvision.transforms as transforms


def get_dataloaders(base_path, batch_size=64) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader,
torch.utils.data.DataLoader, List[str],]:
    """
        Loads training, validation, and test datasets from directory structure and returns DataLoaders.

        Args:
            base_path (str): Path to the dataset directory containing 'train', 'val', and 'test' folders.
            batch_size (int): Number of images per batch.

        Returns:
            trainloader (DataLoader): DataLoader for training data.
            valloader (DataLoader): DataLoader for validation data.
            testloader (DataLoader): DataLoader for test data.
            classes (list): List of class names.
        """

    # Define transformation: Convert to tensor and normalize to [-1, 1] range
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load training dataset
    trainset = torchvision.datasets.ImageFolder(root=os.path.join(base_path, 'train'), transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Load validation dataset
    valset = torchvision.datasets.ImageFolder(root=os.path.join(base_path, 'val'), transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Load test dataset
    testset = torchvision.datasets.ImageFolder(root=os.path.join(base_path, 'test'), transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Return dataloaders and class names
    return trainloader, valloader, testloader, trainset.classes
