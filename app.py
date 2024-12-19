import streamlit as st
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image, ImageDraw
import numpy as np

# Load Models
mtcnn_detector = MTCNN()
# Charger le modèle fine-tuned (ImageNet + ton fine-tuning)
resnet_model = load_model("/Users/aurorekouakou/image_recognition//dataset/models/fine_tuned_resnet50.keras")  

# Title
st.title("Image Recognition")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Function for Face Detection
def detect_faces(img):
    faces = mtcnn_detector.detect_faces(np.array(img))
    return faces

# Function for Object Recognition with Fine-tuned ResNet50
def recognize_objects(img):
    img_resized = img.resize((224, 224))  # Resizing for ResNet50 input
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = resnet_model.predict(img_array)
    return preds  # Custom predictions for your fine-tuned model

# Process uploaded file
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Step 1: Face Detection
    st.subheader("Step 1: Face Detection")
    faces = detect_faces(img)
    if faces:
        draw = ImageDraw.Draw(img)
        for face in faces:
            x, y, width, height = face['box']
            draw.rectangle([x, y, x + width, y + height], outline="red", width=3)
            st.write(f"Face detected with confidence: {face['confidence']:.2f}")
        st.image(img, caption="Image with Detected Faces", use_column_width=True)
    else:
        st.write("No faces detected.")

    # Step 2: Object Recognition with Fine-tuned ResNet50
    st.subheader("Step 2: Object Recognition")
    preds = recognize_objects(img)
    st.write("Predictions (Custom Model):")
    for idx, prob in enumerate(preds[0]):
        st.write(f"Class {idx}: {prob:.2f}")  # Replace with class names if you have a mapping
