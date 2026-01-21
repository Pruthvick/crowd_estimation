import os, cv2, numpy as np, scipy.io as sio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

PART_B = r"C:\Users\HP\crowd_estimation\ShanghaiTech_Crowd_Counting_Dataset\part_B_final\train_data"
IMG_SIZE, EPOCHS, BATCH, LR = 128, 125, 16, 1e-4

def load_data(folder):
    X, y = [], []
    img_dir = os.path.join(folder, "images")
    gt_dir  = os.path.join(folder, "ground_truth")
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

    for file in img_files:
        img_path = os.path.join(img_dir, file)
        gt_path  = os.path.join(gt_dir, "GT_" + file.replace(".jpg", ".mat"))

        img = cv2.imread(img_path)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.0

        mat = sio.loadmat(gt_path)
        pts = mat["image_info"][0][0][0][0][0]
        y.append(len(pts))
        X.append(img)

    return np.array(X)[...,None], np.array(y)

X, y = load_data(PART_B)
print("Loaded Part-B:", X.shape, y.shape)

model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(IMG_SIZE,IMG_SIZE,1)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.3),
    Dense(1)
])

model.compile(optimizer=Adam(LR), loss='mse', metrics=['mae'])
history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH, validation_split=0.2)

model.save("model_B_sparse.h5")
print("💾 Saved model_B_sparse.h5")
