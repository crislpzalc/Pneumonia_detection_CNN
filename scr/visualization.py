"""
visualization.py
================
Utility functions to plot and save:

1. Separate training curves (loss / accuracy)
2. Combined loss-vs-accuracy chart
3. Confusion-matrix image

The figures are stored under ``./figures`` by default so they can be
included in the README or shared in reports.

Author
------
Cristina L. A., June 2025
"""

import matplotlib.pyplot as plt
import os
from sklearn.metrics import ConfusionMatrixDisplay


def plot_training_metrics(train_losses, val_accuracies, save_path='./figures/metrics.png') -> None:
    """Save two side-by-side plots: Training Loss and Validation Accuracy."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))

    # Subplot 1: training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Subplot 2: validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, marker='o', color='green')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_loss_accuracy(train_losses, val_accuracies, save_path='./figures/loss_accuracy_plot.png') -> None:
    """Save a single chart with both loss and accuracy curves."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Training Loss and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_confusion_matrix(cm, class_names, output_path='./figures/confusion_matrix.png') -> None:
    """Save the confusion-matrix image to output_path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='RdPu')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
