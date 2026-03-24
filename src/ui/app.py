import streamlit as st
from pathlib import Path
import yaml
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# App COnfig
parent_path = Path(__file__).resolve().parents[2]
with open(parent_path / 'src/config.yaml', encoding='utf-8') as c:
    config = yaml.safe_load(c)

model_path = parent_path / f"{config['model']['model_directory']}" / 'bcd_yolo.pt'
model_config = config['test']['parameters']

# Load Model
@st.cache_resource
def load_model():
    return YOLO(model_path, task='detect')

# App
st.title("Blood Cell Detection")
st.write("Upload a blood smear image to detect and count RBC, WBC, and Platelets.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    #Convert image to BGR to be used by the model  
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    #RGB image for disply
    image = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)

    # Button outside columns
    run = st.button("Run Detection")

    col_orig, col_pred = st.columns(2)

    with col_orig:
        st.markdown("**Original**")
        st.image(image, use_container_width=True)

    with col_pred:
        right_placeholder = st.empty()

    if run:
        model = load_model()

        with st.spinner("Running inference..."):
            results = model.predict(
                source=np.array(image_bgr),
                iou=0.4,
                imgsz = 640
            )

        result = results[0]
        annotated = cv2.cvtColor(result.plot(),cv2.COLOR_BGR2RGB)

        with right_placeholder.container():
            st.markdown("**Detection Results**")
            st.image(annotated, use_container_width=True)

        # Cell counts
        st.subheader("Cell Counts")
        class_names = model.names
        counts = {name: 0 for name in class_names.values()}
        for cls_idx in result.boxes.cls.tolist():
            counts[class_names[int(cls_idx)]] += 1

        col1, col2, col3 = st.columns(3)
        col1.metric("RBC",       counts.get("RBC", 0))
        col2.metric("WBC",       counts.get("WBC", 0))
        col3.metric("Platelets", counts.get("Platelets", 0))