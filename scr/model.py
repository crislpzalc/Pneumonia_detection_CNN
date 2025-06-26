"""
model.py
========
Convolutional Neural Network (CNN) for binary chest-X-ray classification
(PNEUMONIA vs NORMAL).

Architecture
------------
Input  : 3 × 224 × 224
Conv(3→16, 3×3, BN, ReLU) → MaxPool(2)
Conv(16→32, 3×3, BN, ReLU) → MaxPool(2)
Flatten → FC(32·56·56 → 112) → Dropout(0.5) → ReLU
FC(112 → 84) → Dropout(0.2) → ReLU
FC(84 → 2)   (raw logits)

Notes
-----
* BatchNorm and Dropout switch behaviour automatically via
  ``model.train() / model.eval()``.
* The first fully-connected layer assumes the input image is exactly
  224×224; change to an **AdaptiveAvgPool2d** or ``nn.LazyLinear`` if
  you need variable resolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """Simple CNN with Batch Normalization and Dropout regularisation."""

    def __init__(self) -> None:
        super().__init__()
        # Convolutional block 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Convolutional block 2
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Fully - connected head
        self.fc1 = nn.Linear(32 * 56 * 56, 112)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(112, 84)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(84, 2)

    def forward(self, x) -> torch.Tensor:  # N,C,H,W
        """Forward pass returning raw logits (no softmax)."""
        c1 = self.pool(F.relu(self.bn1(self.conv1(x))))  # N,16,112,112
        c2 = self.pool(F.relu(self.bn2(self.conv2(c1))))  # N,32,56,56
        c2 = torch.flatten(c2, 1)  # N,32*56*56
        f3 = self.dropout1(F.relu(self.fc1(c2)))  # N,112
        f4 = self.dropout2(F.relu(self.fc2(f3)))  # N,84
        out = self.fc3(f4)  # N,2
        return out
