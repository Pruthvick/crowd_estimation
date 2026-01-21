import os
import cv2
import numpy as np
import scipy.io as sio
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
MODEL_PATH = "model_B_sparse.h5"   # change if filename differs
PART_B_TEST_IMG = r"C:\Users\HP\crowd_estimation\ShanghaiTech_Crowd_Counting_Dataset\part_B_final\test_data\images"
PART_B_TEST_GT  = r"C:\Users\HP\crowd_estimation\ShanghaiTech_Crowd_Counting_Dataset\part_B_final\test_data\ground_truth"

# ---------------- LOAD MODEL ----------------
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MSE

model = load_model(
    MODEL_PATH,
    custom_objects={"mse": MeanSquaredError(), "MeanSquaredError": MSE()}
)


# ---------------- LOAD TEST DATA ----------------
y_true, y_pred = [], []
img_files = sorted([f for f in os.listdir(PART_B_TEST_IMG) if f.endswith(".jpg")])

print(f"📂 Found {len(img_files)} test images")

for f in img_files:
    img_path = os.path.join(PART_B_TEST_IMG, f)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    x = np.expand_dims(img, axis=(0, -1))

    # Predict
    pred = model.predict(x)[0][0]

    # Load GT
    mat_path = os.path.join(PART_B_TEST_GT, "GT_" + f.replace(".jpg", ".mat"))
    mat = sio.loadmat(mat_path)
    ann_points = mat["image_info"][0][0][0][0][0]
    true_count = len(ann_points)

    y_true.append(true_count)
    y_pred.append(pred)

# ---------------- METRICS ----------------
mae = mean_absolute_error(y_true, y_pred)
rmse = math.sqrt(mean_squared_error(y_true, y_pred))

print("\n📌 TEST RESULTS (PART-B)")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")

# ---------------- PLOTS ----------------
# True vs Pred
plt.figure(figsize=(6,4))
plt.plot(y_true, label="True Count")
plt.plot(y_pred, label="Predicted Count")
plt.title("True vs Predicted Count - Part-B")
plt.xlabel("Image Index")
plt.ylabel("People Count")
plt.legend()
plt.grid(True)
plt.savefig("B_true_vs_pred.png")
plt.show()

# Error curve
errors = np.abs(np.array(y_true) - np.array(y_pred))
plt.figure(figsize=(6,4))
plt.plot(errors)
plt.title("Absolute Error per Test Image - Part-B")
plt.xlabel("Image Index")
plt.ylabel("Error")
plt.grid(True)
plt.savefig("B_error_curve.png")
plt.show()

# Error histogram
plt.figure(figsize=(6,4))
plt.hist(errors, bins=20)
plt.title("Error Distribution - Part-B")
plt.xlabel("Error Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("B_error_hist.png")
plt.show()

print("\n📁 Graphs saved:")
print(" - B_true_vs_pred.png")
print(" - B_error_curve.png")
print(" - B_error_hist.png")
