import streamlit as st
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from HOG import HOG
from Softmax import Softmax

# Set up the title
st.title("🚀 HOG with Softmax for Classification of Cars, Planes, Trains")

# Description and instructions
st.markdown("""
    <p style="text-align: center; font-size: 18px; color: #4c4c4c;">
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

# Model URL and file path
REPO_URL = "ibrahimsultan/HOG_Softmax_image_classifier"  # Hugging Face repository
MODEL_FILE = "softmax_HOG.pth"  # Model file name

# Initialize the placeholder for downloading message
download_message = st.empty()

# Path to download the model
with download_message.container():
    st.markdown("""
        <div style="text-align: center; font-size: 16px; color: #0000ff;">
            <p>Downloading the model... Please wait.</p>
        </div>
    """, unsafe_allow_html=True)

    # Download the model
    downloaded_model_path = hf_hub_download(
        repo_id=REPO_URL,
        filename=MODEL_FILE  # Model file you want to download
    )

# Remove the downloading message after the model is downloaded
download_message.empty()

# Load the model after downloading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Softmax(inputNeurons=8100, hlayer1=2100, outputs=3).to(device)
model.load_state_dict(torch.load(downloaded_model_path, map_location=device))
model.eval()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')

    hog = HOG(9, 8, 2)
    features = hog.GetFeatures(image)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

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

    # Display the uploaded image in a larger size
    st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)
else:
    st.markdown("""
        <div style="text-align: center; font-size: 16px; color: #a6a6a6;">
            Please upload an image to see the prediction.
        </div>
    """, unsafe_allow_html=True)
