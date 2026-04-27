import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import gdown
import os

CLASS_NAMES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
               'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
               'River', 'SeaLake']

PIXEL_MAX = 3500.0

# Replace this with your actual Google Drive file ID
GDRIVE_FILE_ID = "1-SDqpNG5o9x9i9ulcAupZML2v_YQ2Iup"

@st.cache_resource
def load_model():
    if not os.path.exists('best_model.pth'):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, 'best_model.pth', quiet=False)

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load('best_model.pth',
                                      map_location='cpu'))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

st.title("🛰️ Satellite Scene Classifier")
st.write("Upload a satellite image patch to classify its land-use type.")

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB").resize((64, 64))
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded patch", use_column_width=True)

    with col2:
        img_array = np.array(img).astype(float)
        img_tensor = torch.tensor(img_array).permute(2, 0, 1).float()
        img_tensor = img_tensor / PIXEL_MAX
        img_tensor = transform(img_tensor).unsqueeze(0)

        model = load_model()
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]

        st.subheader("Predictions")
        top3_probs, top3_idx = probs.topk(3)
        for prob, idx in zip(top3_probs, top3_idx):
            st.progress(float(prob),
                        text=f"{CLASS_NAMES[idx]}: {prob:.1%}")

        st.success(f"Top prediction: **{CLASS_NAMES[probs.argmax()]}**")
