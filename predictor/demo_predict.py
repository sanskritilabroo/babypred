import torch
import torch.nn as nn
from torchvision import models
import cv2
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
import numpy as np

# ----------------------------
# 1️⃣ Device setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 2️⃣ Define and load trained model
# ----------------------------
model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model = model.to(device)

# ✅ Load trained weights
checkpoint_path = "C:/Users/ASUS PC/Desktop/model/keras/trained_model.pth"  # <-- change this path
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ----------------------------
# 3️⃣ Define preprocessing
# ----------------------------
def get_preprocessing():
    return Compose([
        Resize(640, 640),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

preprocess = get_preprocessing()

# ----------------------------
# 4️⃣ Load and preprocess input image
# ----------------------------
image_path = "results.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

augmented = preprocess(image=image_rgb)
input_tensor = augmented["image"].unsqueeze(0).to(device)

# ----------------------------
# 5️⃣ Inference
# ----------------------------

with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    pred_class = predicted.item()

# ----------------------------
# 6️⃣ Class mapping
# ----------------------------
class_names = {0: "unsafe_baby_on_back", 1: "safe_baby_on_stomach"}
pred_label = class_names[pred_class]

display_image = cv2.resize(image, (640, 640))
cv2.putText(display_image, f"Predicted: {pred_label}", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

cv2.imshow("Prediction", display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"✅ Predicted class: {pred_label}")
