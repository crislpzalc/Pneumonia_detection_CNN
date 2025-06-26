"""
train.py
========
Training loop with early stopping, CSV logging and metric plotting
for the Pneumonia-vs-Normal CNN.

Functions
---------
train_model(...)
    Trains the model, saves the best weights, logs metrics to a CSV file
    and plots the loss / accuracy curves.

Author
------
Cristina L. A., June 2025
"""

from typing import Tuple, List
import torch
import pandas as pd
import os
from evaluate import evaluate
from visualization import plot_loss_accuracy


def train_model(model, trainloader, valloader, device, optimizer, criterion, epochs, save_path,
                classes, patience=3) -> Tuple[List[float], List[float]]:
    """
        Train a deep learning model with early stopping, logging and visualization.

        Args:
            model (torch.nn.Module): The model to train.
            trainloader (DataLoader): Dataloader for training data.
            valloader (DataLoader): Dataloader for validation data.
            device (torch.device): Device to run the training on (CPU or GPU).
            optimizer (torch.optim.Optimizer): Optimizer used for model training.
            criterion (torch.nn.Module): Loss function.
            epochs (int): Maximum number of training epochs.
            save_path (str): File path to save the best model.
            classes (List[str]): Class names, passed to evaluation.
            patience (int, optional): Number of epochs with no improvement to wait before early stopping. Defaults to 3.

        Returns:
            Tuple[List[float], List[float]]: Lists of training losses and validation accuracies per epoch.
        """

    # Defensive input validation
    if epochs < 1:
        raise ValueError("`epochs` must be >= 1.")
    if patience < 1:
        raise ValueError("`patience` must be >= 1.")
    if not isinstance(save_path, str):
        raise TypeError("`save_path` must be a string.")

    train_losses = []
    val_accuracies = []
    best_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()               # Clear previous gradients
            outputs = model(inputs)             # Forward pass
            loss = criterion(outputs, labels)   # Compute loss
            loss.backward()                     # Backpropagation
            optimizer.step()                    # Update model weights

            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

        # Validation step
        val_accuracy = evaluate(model, valloader, device, classes, print_output=False)
        val_accuracies.append(val_accuracy)

        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
            try:
                torch.save(model.state_dict(), save_path)
                print(f"Epoch {epoch + 1}: Best saved model with val_accuracy = {val_accuracy:.4f}")
            except IOError as e:
                print(f"IOError occurred: {e}")
        else:
            epochs_without_improvement += 1
            # Trigger early stopping if no improvement for 'patience' epochs
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs. No improvement in last {patience}.")
                break

    # Save training log as CSV
    os.makedirs('./figures', exist_ok=True)
    try:
        df = pd.DataFrame({
            'epoch': list(range(1, len(train_losses) + 1)),
            'train_loss': train_losses,
            'val_accuracy': val_accuracies
        })
        df.to_csv('./figures/training_log.csv', index=False)
        print("Saved logs in figures/training_log.csv")
    except IOError as e:
        print(f"IOError occurred: {e}")

    # Plot metrics
    try:
        plot_loss_accuracy(train_losses, val_accuracies)
    except IOError as e:
        print(f"IOError occurred: {e}")

    return train_losses, val_accuracies
