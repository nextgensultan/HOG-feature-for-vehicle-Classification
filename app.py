import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
from HOG import HOG
from Softmax import Softmax

# Set up the title
st.title("🚀 HOG Feature Extraction and Classification")

# Description and instructions
st.markdown("""
    <p style="text-align: center; font-size: 16px; color: #4c4c4c;">
    Upload an image to classify it into one of the predefined categories. 
    Supported formats: <b>JPG, PNG, JPEG, BMP</b>.
    </p>
    """, unsafe_allow_html=True)

# Custom file uploader with an enhanced drag-and-drop area
st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h3>Select or drag an image:</h3>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png", "bmp"],
    label_visibility="collapsed",
    help="Drag and drop or click to upload an image."
)

if uploaded_file is not None:
    # Load the image and convert to grayscale
    image = Image.open(uploaded_file).convert('L')

    # Initialize HOG feature extractor
    hog = HOG(9, 8, 2)
    features = hog.GetFeatures(image)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Softmax(inputNeurons=8100, hlayer1=2100, outputs=3).to(device)
    model.load_state_dict(torch.load("softmax_HOG.pth", map_location=device))
    model.eval()

    # Make a prediction
    with torch.no_grad():
        features_tensor = features_tensor.to(device)
        output = model(features_tensor)
        prediction = torch.argmax(output, dim=1).item()

    # Display the predicted class
    labels = ["cars", "planes", "trains"]
    classMap = {idx: label for idx, label in enumerate(labels)}
    st.markdown(f"""
        <div style="text-align: center; margin: 20px 0;">
            <h2 style="color: #2e8b57;">Predicted class: {classMap[prediction]}</h2>
        </div>
    """, unsafe_allow_html=True)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True, output_format="PNG")
else:
    st.markdown("""
        <div style="text-align: center; font-size: 16px; color: #a6a6a6;">
            Please upload an image to see the prediction.
        </div>
    """, unsafe_allow_html=True)
