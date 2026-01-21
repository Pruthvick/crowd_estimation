import os
import cv2
import numpy as np
import scipy.io as sio
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import matplotlib.pyplot as plt

# ------------- CONFIG -------------
MODEL_PATH = "model_A_dense.h5"
IMG_DIR = r"C:\Users\HP\crowd_estimation\ShanghaiTech_Crowd_Counting_Dataset\part_A_final\test_data\images"
GT_DIR  = r"C:\Users\HP\crowd_estimation\ShanghaiTech_Crowd_Counting_Dataset\part_A_final\test_data\ground_truth"

# ------------- LOAD MODEL -------------
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MSE

model = load_model(
    MODEL_PATH,
    custom_objects={"mse": MeanSquaredError(), "MeanSquaredError": MSE()}
)


# ------------- TEST LOOP -------------
y_true, y_pred = [], []

img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])

for f in img_files:
    img_path = os.path.join(IMG_DIR, f)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    x = np.expand_dims(img, axis=(0, -1))

    # Predict
    pred = model.predict(x)[0][0]

    # Load true count
    mat_path = os.path.join(GT_DIR, "GT_" + f.replace(".jpg", ".mat"))
    mat = sio.loadmat(mat_path)
    ann_points = mat["image_info"][0][0][0][0][0]
    true_count = len(ann_points)

    y_true.append(true_count)
    y_pred.append(pred)

# ------------- METRICS -------------
mae = mean_absolute_error(y_true, y_pred)
rmse = math.sqrt(mean_squared_error(y_true, y_pred))

print("\n📌 TEST RESULTS")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")

# ------------- PLOTS -------------
plt.figure(figsize=(6,4))
plt.plot(y_true, label="True Count")
plt.plot(y_pred, label="Predicted Count")
plt.title("True vs Predicted Count (Test Set)")
plt.xlabel("Test Images Index")
plt.ylabel("People Count")
plt.legend()
plt.grid(True)
plt.savefig("plot_true_vs_pred.png")
plt.show()

errors = np.abs(np.array(y_true) - np.array(y_pred))

plt.figure(figsize=(6,4))
plt.plot(errors)
plt.title("Absolute Error per Test Image")
plt.xlabel("Test Index")
plt.ylabel("Error")
plt.grid(True)
plt.savefig("plot_error_curve.png")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(errors, bins=20)
plt.title("Distribution of Errors")
plt.xlabel("Error Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("plot_error_hist.png")
plt.show()

print("\n📁 Graphs saved as:")
print(" - plot_true_vs_pred.png")
print(" - plot_error_curve.png")
print(" - plot_error_hist.png")
