"""
evaluate.py
===========
Validation / testing helper for the Pneumonia-vs-Normal CNN.

Functions
---------
evaluate(...)
    Runs the model on a DataLoader, computes accuracy and—optionally—
    prints and saves the confusion matrix plus a classification report.

Author
------
Cristina L. A., June 2025
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch
from visualization import save_confusion_matrix


def evaluate(model, dataloader, device, class_names, threshold=0.5, print_output=True)\
        -> float:
    """Evaluate a trained model on validation or test data.

    Parameters
    ----------
    model : nn.Module
        Trained model in *eval* mode.
    dataloader : DataLoader
        Validation or test DataLoader.
    device : torch.device
        CPU or CUDA device.
    class_names : list[str]
        List with readable class labels (for plotting).
    threshold : float, default = 0.5
        Decision threshold applied to the positive-class probability.
    print_output : bool, default = True
        Whether to print metrics and display the confusion matrix.

    Returns
    -------
    accuracy : float
        Overall accuracy of the predictions.
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            pneumonia_probs = probs[:, 1]

            # Store results
            all_probs.extend(pneumonia_probs.cpu().numpy())
            predicted = (pneumonia_probs > threshold).long()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print("Accuracy: ", accuracy)

    if print_output:
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues')
        print("\nClassification Report: ")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        # Save PNG to ./figures
        save_confusion_matrix(cm, class_names)

    return accuracy
