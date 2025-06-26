"""
main.py
=======
Main script to run the full pneumonia detection pipeline:
- Downloads and prepares the dataset
- Initializes and trains the CNN model
- Evaluates the model on the test set
- Saves training metrics and model summary

Usage
-----
python src/main.py --data_dir ./data --model_dir ./models --epochs 20 --batch_size 32 --lr 0.001 --threshold 0.5

Author
------
Cristina L. A., June 2025
"""

import torch
from model import Net
from dataLoader import get_dataloaders
from train import train_model
from evaluate import evaluate
import torch.nn as nn
from prepareData import prepare_and_split_data
import os
import argparse
from torchinfo import summary
from visualization import plot_training_metrics

# Argument parser for flexible training configuration
parser = argparse.ArgumentParser(description="src Detection Training Pipeline")
parser.add_argument('--data_dir', type=str, default='./data', help='Path to dataset')
parser.add_argument('--model_dir', type=str, default='./models', help='Path to save models')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold')
args = parser.parse_args()

if __name__ == '__main__':
    # Step 1: Download and split dataset
    prepare_and_split_data(args.data_dir)

    # Step 2: Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)

    # Step 3: Initialize model, loss function, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # Step 4: Load DataLoaders
    base_path = args.data_dir
    trainloader, valloader, testloader, classes = get_dataloaders(base_path)

    # Step 5: Train model with early stopping
    os.makedirs(args.model_dir, exist_ok=True)
    best_model_path = os.path.join(args.model_dir, 'best_model.pt')

    train_losses, val_accuracies = train_model(model, trainloader, valloader, device, optimizer, criterion,
                                               args.epochs, best_model_path, classes)

    # Step 6: Save plots and training log
    plot_training_metrics(train_losses, val_accuracies)

    # Step 7: Save model summary to file
    os.makedirs('./figures', exist_ok=True)
    with open('./figures/model_summary.txt', 'w', encoding='utf-8') as f:
        f.write(str(summary(model, input_size=(args.batch_size, 3, 224, 224), device=device)))

    # Step 8: Load best model and evaluate on test set
    model.load_state_dict(torch.load(best_model_path))
    evaluate(model, testloader, device, classes, args.threshold)
