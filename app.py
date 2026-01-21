import os
from flask import Flask, render_template, request
import numpy as np
import cv2
from PIL import Image

# ==== Torch Optical Flow ====
import torch
import torch.nn as nn
from torchvision import transforms

# ==== Keras Density ====
from tensorflow.keras.models import load_model
from scipy.ndimage import gaussian_filter

# ==========================================
# CONFIG
# ==========================================
UPLOAD_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = 128
FLOW_MODEL_PATH = "best_crowdflow_model.pth"
DENSE_MODEL_PATH = "model_A_dense.h5"
SPARSE_MODEL_PATH = "model_B_sparse.h5"
SPARSE_THRESHOLD = 50   # auto-switch cut-off

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER



# Load Density Models 
# ==========================================
from tensorflow.keras.losses import MeanSquaredError

custom_objects = {"mse": MeanSquaredError(), "MeanSquaredError": MeanSquaredError()}

model_dense = load_model(DENSE_MODEL_PATH, custom_objects=custom_objects, compile=False)
model_sparse = load_model(SPARSE_MODEL_PATH, custom_objects=custom_objects, compile=False)


# ==========================================
# Optical Flow Model (PyTorch)
# ==========================================
class SimpleFlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 2, 4, 2, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

flow_model = SimpleFlowNet()
flow_model.load_state_dict(torch.load(FLOW_MODEL_PATH, map_location="cpu"))
flow_model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])


# ==========================================
# Heatmap Generator
# ==========================================
def generate_heatmap(img, count):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    hmap = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    for _ in range(min(int(count), 150)):
        y, x = np.random.randint(0, IMG_SIZE), np.random.randint(0, IMG_SIZE)
        hmap[y, x] += 1.0

    hmap = gaussian_filter(hmap, sigma=4)
    norm = (hmap / (hmap.max() + 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


# ==========================================
# Predict Flow Direction
# ==========================================
def predict_flow(img: Image.Image):
    tensor = transform(img).unsqueeze(0).float()
    with torch.no_grad():
        flow = flow_model(tensor)
    flow = flow.squeeze().permute(1, 2, 0).numpy()

    avg_dx = np.mean(flow[..., 0])
    avg_dy = np.mean(flow[..., 1])
    angle = np.arctan2(avg_dy, avg_dx)

    # map angle to text
    dirs = [
        ("Right", (-np.pi/8, np.pi/8)),
        ("Up-Right", (np.pi/8, 3*np.pi/8)),
        ("Up", (3*np.pi/8, 5*np.pi/8)),
        ("Up-Left", (5*np.pi/8, 7*np.pi/8)),
        ("Left", (7*np.pi/8, np.pi)),
        ("Left", (-np.pi, -7*np.pi/8)),
        ("Down-Left", (-7*np.pi/8, -5*np.pi/8)),
        ("Down", (-5*np.pi/8, -3*np.pi/8)),
        ("Down-Right", (-3*np.pi/8, -np.pi/8)),
    ]
    direction = "Unknown"
    for name, (low, high) in dirs:
        if low <= angle <= high:
            direction = name
            break

    # Draw arrow
    img_cv = np.array(img.resize((IMG_SIZE, IMG_SIZE)))[:, :, ::-1].copy()
    center = (IMG_SIZE // 2, IMG_SIZE // 2)
    tip = (int(center[0] + avg_dx * 30), int(center[1] + avg_dy * 30))
    cv2.arrowedLine(img_cv, center, tip, (0, 255, 0), 2, tipLength=0.3)

    return img_cv, direction


# ==========================================
# ROUTES
# ==========================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return "No file!", 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    img = Image.open(filepath).convert("RGB")

    # ---- Density Prediction ----
    gray = img.resize((IMG_SIZE, IMG_SIZE)).convert("L")
    arr = np.expand_dims(np.array(gray) / 255.0, (0, -1))

    sparse_pred = model_sparse.predict(arr)[0][0]
    if sparse_pred < SPARSE_THRESHOLD:
        final_count = sparse_pred
        selected_model = "Sparse Model (Part-B)"
    else:
        final_count = model_dense.predict(arr)[0][0]
        selected_model = "Dense Model (Part-A)"

    final_count = final_count = max(0, int(final_count))

    # ---- Heatmap ----
    heatmap = generate_heatmap(img, final_count)
    heatpath = os.path.join(app.config["UPLOAD_FOLDER"], "heat_" + file.filename)
    cv2.imwrite(heatpath, heatmap)

    # ---- Flow ----
    flow_img, direction = predict_flow(img)
    flowpath = os.path.join(app.config["UPLOAD_FOLDER"], "flow_" + file.filename)
    cv2.imwrite(flowpath, flow_img)

    return render_template("index.html",
                           uploaded=filepath,
                           heatmap=heatpath,
                           flow=flowpath,
                           count=round(final_count, 1),
                           model=selected_model,
                           direction=direction)


if __name__ == "__main__":
    app.run(debug=True)
