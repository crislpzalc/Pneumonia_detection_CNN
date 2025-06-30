import matplotlib.pyplot as plt
import numpy as np, torch
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image


# Define CNN
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

# Load pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

transform = T.Compose([T.Resize((224,224)),
                       T.ToTensor(),
                       T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

# Upload and visualize an image
def predict_gradcam(image):
    # prediction
    img  = image.convert("RGB")
    plt.imshow(image); plt.axis('off'); plt.show()
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(tensor), dim=1)[0,1].item()
    prob= f"{p:.3f}"
    label= f"{'PNEUMONIA' if p>0.5 else 'NORMAL'}"

    # Grad-CAM
    target_layer = model.conv2
    input_tensor = transform(img).unsqueeze(0).to(device)
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    img_np = np.array(img.resize((224,224)), dtype=np.float32)/255.0
    heatmap = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    heatmap_pil = Image.fromarray(heatmap)

    return prob, label, heatmap_pil

demo = gr.Interface(
    fn=predict_gradcam,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray"),
    outputs=[gr.Textbox(label="Probability of Pneumonia"), gr.Label(label="Prediction"), gr.Image(label="Grad-CAM")],
    title="🫁 Pneumonia Detection from Chest X-rays",
    description="Upload a chest X-ray to see whether it shows signs of pneumonia. The model will predict the probability and show a Grad-CAM visualization of the most important regions.",
    flagging_mode="never"
)

demo.launch()
