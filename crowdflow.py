import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

# ========================================
# CONFIG
# ========================================
BASE_DIR = r"C:\Users\HP\crowd_behaviour\TUBCrowdFlow\TUBCrowdFlow"
BATCH_SIZE = 2
EPOCHS = 50              
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("📌 Using device:", DEVICE)

# ========================================
# .flo Reader
# ========================================
def read_flo_file(filename):
    """Reads a .flo optical flow file"""
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            raise Exception('Invalid .flo file: {}'.format(filename))
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    return np.resize(data, (h, w, 2))

# ========================================
# Dataset Class
# ========================================
class CrowdFlowDataset(Dataset):
    def __init__(self, base_dir):
        # Load ALL IM folders automatically
        self.image_paths = sorted(glob.glob(os.path.join(base_dir, "images", "*", "*.png")))
        self.flow_paths  = sorted(glob.glob(os.path.join(base_dir, "gt_flow", "*", "*.flo")))

        min_len = min(len(self.image_paths), len(self.flow_paths))
        self.image_paths = self.image_paths[:min_len]
        self.flow_paths  = self.flow_paths[:min_len]

        print(f"📂 Found {len(self.image_paths)} image-flow pairs")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # ----- Load input frame -----
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)

        # ----- Load GT flow -----
        flow = read_flo_file(self.flow_paths[idx])
        flow = cv2.resize(flow, (128, 128))
        flow = torch.tensor(flow, dtype=torch.float32).permute(2, 0, 1)

        return img, flow

# ========================================
# Model
# ========================================
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

# ========================================
# TRAINING
# ========================================
def train():
    dataset = CrowdFlowDataset(BASE_DIR)
    if len(dataset) == 0:
        print("❌ ERROR: No dataset found. Check BASE_DIR path")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleFlowNet().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_loss = float("inf")
    print(f"🚀 Training on {len(dataset)} samples...")

    for epoch in range(EPOCHS):
        total_loss = 0.0
        model.train()

        for imgs, flows in dataloader:
            imgs, flows = imgs.to(DEVICE), flows.to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, flows)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"📊 Epoch [{epoch+1}/{EPOCHS}] ➝ Loss = {avg_loss:.6f}")

        # Save Best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_crowdflow_model.pth")
            print("💾 Saved BEST model ✔")

    # Save final model
    torch.save(model.state_dict(), "crowdflow_model_last.pth")
    print("🏁 Done. Best model = best_crowdflow_model.pth")

# ========================================
# RUN
# ========================================
if __name__ == "__main__":
    train()
