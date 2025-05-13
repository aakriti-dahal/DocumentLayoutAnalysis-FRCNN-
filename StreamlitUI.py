import streamlit as st
from PIL import Image
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

# Load model
@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 6  # 5 classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load("Document Analysis.pth", map_location=torch.device('cpu')))
    model.to("cpu")
    model.eval()
    return model

model = load_model()

# Category name map
CATEGORY_NAMES = {
    1: "image",
    2: "paragraph",
    3: "table",
    4: "title"
}

# Streamlit UI
st.title("Document Layout Detection")
st.write("Upload a document image to detect layout components like images, paragraphs, tables, and titles.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_tensor = ToTensor()(img).unsqueeze(0)

    with torch.no_grad():
        prediction = model(img_tensor)

    boxes = prediction[0]['boxes'].numpy()
    labels = prediction[0]['labels'].numpy()
    scores = prediction[0]['scores'].numpy()

    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(np.array(img))
    confidence_threshold = 0.5

    for box, label, score in zip(boxes, labels, scores):
        if score > confidence_threshold:
            label_name = CATEGORY_NAMES.get(label, "unknown")
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            label_text = f"{label_name} ({score:.2f})"
            ax.text(x_min, y_min - 5, label_text, color='red', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.5))

    ax.axis('off')
    st.pyplot(fig)
