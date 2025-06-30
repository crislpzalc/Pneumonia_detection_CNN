# Pneumonia Detection with a Convolutional Neural Network (PyTorch)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/Try%20Demo-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/spaces/tu_usuario/tu_space)
[![Open In Colab](https://img.shields.io/badge/%20Open%20in%20Colab-grey?logo=googlecolab&logoColor=orange&labelColor=grey&color=ffa500)](https://colab.research.google.com/drive/1SZy7mXp8qbPJ6tY9aCd-dwvn5efIX2SP?usp=sharing)


A complete, reproducible pipeline to detect **pneumonia** from chest-X-ray images using a compact Convolutional Neural Network built in PyTorch.  
The project covers data preparation, training with early stopping, metric logging, visualisation, threshold tuning and inference.

---

## Highlights

| Feature | Description |
|---------|-------------|
| **End-to-end code** | Download → split → train → evaluate → plots → logs. |
| **Early Stopping**  | Stops when validation accuracy stops improving. |
| **Regularisation**  | Batch Normalisation & Dropout. |
| **Threshold tuning** | Trade-off between recall & precision (important in medicine). |
| **Google Colab demo** | Run the model in one click, no local setup. |

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Live Web Demo in Hugging Face](#live-web-demo-in-hugging-face)
3. [Google Colab Demo](#google-colab-demo)  
4. [Dataset](#dataset)  
5. [Model Architecture](#model-architecture)  
6. [Training & Evaluation Results](#training--evaluation-results)  
7. [Project Structure](#project-structure)  
8. [Future Work](#future-work)  
9. [License](#license)  

---

## Quick Start

### 1.  Clone the repo
````
git clone https://github.com/crislpzalc/Pneumonia_detection_CNN.git
cd Pneumonia_detection_CNN
````
### 2.  Install dependencies
````
pip install -r requirements.txt
````
### 3.  Run the full training pipeline (downloads ≈190 MB)
````
python src/main.py \
    --data_dir ./data \
    --model_dir ./models \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.001
````

> **Tip:** a free GPU on Google Colab or Kaggle speeds training dramatically.

---
## Live Web Demo in Hugging Face

[![Hugging Face](https://img.shields.io/badge/Try%20Demo-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/spaces/tu_usuario/tu_space)

You can now test the model **directly in the browser** — no setup, no coding — thanks to [Gradio](https://www.gradio.app/) and [Hugging Face Spaces](https://huggingface.co/spaces).

Upload any chest X-ray image and receive:
- A pneumonia vs. normal prediction
- A probability score
- A Grad-CAM heatmap showing model attention

**Try it live**: [huggingface.co/spaces/CristinaLA/pneumonia-detector](https://huggingface.co/spaces/CristinaLA/pneumonia-detector)

Code for the demo is available in [`demo_huggingface/`](./demo_huggingface).

## Google Colab Demo

Run live inference with the trained model and sample X-rays:

[![Open In Colab](https://img.shields.io/badge/%20Open%20in%20Colab-grey?logo=googlecolab&logoColor=orange&labelColor=grey&color=ffa500)](https://colab.research.google.com/drive/1SZy7mXp8qbPJ6tY9aCd-dwvn5efIX2SP?usp=sharing)


The notebook:

1. Clones this repository.
2. Downloads `best_model.pt`.
3. Runs predictions on two sample X-ray images (one NORMAL, one PNEUMONIA).
4. Displays probabilities and predicted label.

---

## Dataset

| Property               | Value                                                        |
| ---------------------- | ------------------------------------------------------------ |
| **Source**             | [Kaggle – “Neumona X-rays dataset”](https://www.kaggle.com/datasets/gonzajl/neumona-x-rays-dataset) |
| **Images**             | 6 298 (≈ 50 % Pneumonia / 50 % Normal)                       |
| **Train / Val / Test** | 60 % / 20 % / 20 %                                           |
| **Pre-processing**     | Resize → 224 × 224 px  •  Normalise to $-1, 1 $              |

> ⚠️ Due to size constraints, data is not included in the repository.
---

## Model Architecture

```
Input : 3 × 224 × 224
└─ Conv(3→16, 3×3) + BatchNorm + ReLU
   └─ MaxPool(2)
      └─ Conv(16→32, 3×3) + BatchNorm + ReLU
         └─ MaxPool(2)
            └─ Flatten
               └─ Linear(100352 → 112) + Dropout(0.5) + ReLU
                  └─ Linear(112 → 84)   + Dropout(0.2) + ReLU
                     └─ Linear(84  → 2 logits)
Total parameters ≈ 11.3 M
```

Full parameter summary saved to **`figures/model_summary.txt`**.

---

## Training & Evaluation Results

| Metric (threshold = 0.5)     | Value     |
| ---------------------------- | --------- |
| **Best Validation Accuracy** | **0.960** |
| **Test Accuracy**            | **0.968** |
| **Test Precision (Pos.)**    | **0.98**  |
| **Test Recall (Pos.)**       | **0.96**  |
| **F1-score (macro)**         | **0.97**  |
| **AUC-ROC**                  | **0.99**  |

<p align="center">
  <img src="figures/metrics.png" width="420">
  <img src="figures/confusion_matrix.png" width="420">
</p>

### Threshold discussion

The default decision threshold (0.5) yields **46 false negatives** (patients with pneumonia predicted as normal).
In a clinical context we favour **higher recall** to minimise missed cases. Lowering the threshold (e.g. 0.25) eliminates false negatives at the cost of more false positives. The Colab notebook shows how to experiment with different thresholds and visualise precision/recall trade-offs.

---

## Project Structure

```
pneumonia-cnn-pytorch
├── src/
│   ├── main.py           # Entry-point script (argparse)
│   ├── model.py          # CNN definition
│   ├── train.py          # Training loop + early stopping
│   ├── evaluate.py       # Validation & test evaluation
│   ├── dataLoader.py     # DataLoaders
│   ├── prepareData.py    # Download & split dataset
│   └── visualization.py  # Plotting utilities
│
├── models/               # best_model.pt (generated)
├── figures/              # plots, confusion matrix, model summary, training_log.csv
├── data/                  
├── requirements.txt
└── README.md
```
> ⚠️ Due to file size limits on GitHub, the trained model (`best_model.pt`) is available through the [Google Colab Demo](https://colab.research.google.com/drive/1SZy7mXp8qbPJ6tY9aCd-dwvn5efIX2SP?usp=sharing)
---

## Future Work

* Apply data augmentation (random flips, rotations, brightness jitter).
* Compare against a **ResNet-18** transfer-learning baseline.
* Hyper-parameter sweep with Optuna (learning-rate, batch size, optimiser).

---

## License

This project is licensed under the [MIT License](LICENSE).

You are free to use, modify, and distribute this code with proper attribution.

---

> **Author:** [Cristina L. A.](https://www.linkedin.com/in/cristina-lopez-alcazar/), June 2025
