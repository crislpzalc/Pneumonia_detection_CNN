# 🫁 Hugging Face Demo – Pneumonia Detection

This folder contains the code used to deploy the interactive pneumonia detection demo on [Hugging Face Spaces](https://huggingface.co/spaces/CristinaLA/pneumonia-detector)

The demo allows users to upload a chest X-ray and receive:
- A prediction (**Pneumonia** or **Normal**)
- A probability score
- A Grad-CAM heatmap highlighting relevant lung regions

The interface is built using **Gradio**, and the model is a simple CNN trained on the [Chest X-ray Pneumonia dataset](https://www.kaggle.com/datasets/gonzajl/neumona-x-rays-dataset).

---

## Files

- `app.py` → the Gradio app
- `requirements.txt` → libraries used in the Hugging Face Space
- `best_model.pt` → stored on the Hugging Face server (not included here)

---

## Try it live

You can try the model online without any setup: [Live demo on Hugging Face](https://huggingface.co/spaces/CristinaLA/pneumonia-detector)

---

## Note

This version is optimized for deployment.  
For training, data exploration and other utilities, see the main folders in this repository.

