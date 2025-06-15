import streamlit as st
import numpy as np
import cv2
import os
from src.utils import load_pickle_model, load_cnn_model, predict

# Load trained models
svm_model = load_pickle_model("models/svm_model.pkl")
knn_model = load_pickle_model("models/knn_model.pkl")
nb_model = load_pickle_model("models/nb_model.pkl")
dt_model = load_pickle_model("models/dt_model.pkl")
cnn_model = load_cnn_model("models/cnn_model.h5")

def predict_cnn(image, model):
    """Predict using the CNN model."""
    # Convert the image to grayscale if it is in color
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.resize(image, (64, 64)) / 255.0
    processed_image = np.expand_dims(processed_image, axis=(0, -1))  # Add batch and channel dimensions
    prediction = model.predict(processed_image)
    return "NO Tumor" if np.argmax(prediction) == 0 else "Brain Tumor Detected"

def run_app():
    """Run the Streamlit app."""
    st.title("🧠 Brain Tumor Detection Using Machine Learning")
    st.write("Upload a Brain MRI image to check for tumors.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model_choice = st.selectbox("Choose a model", ["SVM", "KNN", "Naïve Bayes", "Decision Tree", "CNN"])
    if st.button("Predict"):
        with st.spinner("Processing..."):
            if model_choice == "SVM":
                result = predict(image, svm_model, "SVM")
            elif model_choice == "KNN":
                result = predict(image, knn_model, "KNN")
            elif model_choice == "Naïve Bayes":
                result = predict(image, nb_model, "Naïve Bayes")
            elif model_choice == "Decision Tree":
                result = predict(image, dt_model, "Decision Tree")
            else:  # CNN
                result = predict_cnn(image, cnn_model)
    
        st.success(f"Prediction: {result}")

if __name__ == "__main__":
    run_app()
