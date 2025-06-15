import os
import numpy as np
import cv2
import pickle
import streamlit as st
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.utils import to_categorical #type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense #type: ignore

def preprocess_image(image, target_size=(64, 64), flatten=False):
    """Preprocess an image by resizing, normalizing, and optionally flattening."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, target_size) / 255.0
    return image.flatten().reshape(1, -1) if flatten else np.expand_dims(image, axis=(0, -1))

def load_pickle_model(model_path):
    """Load a model saved as a pickle file."""
    try:
        with open(model_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def load_cnn_model(model_path):
    """Load a CNN model saved in HDF5 format."""
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Error loading CNN model: {e}")
        return None

def predict(image, model, model_type):
    """Predict the class of an image using the specified model."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.resize(image, (64, 64)) / 255.0
    processed_image = processed_image.flatten().reshape(1, -1)
    result = model.predict(processed_image)[0]
    return "NO Tumor" if result == 0 else "Brain Tumor Detected"

def load_data(dataset_path="src/Dataset"):
    """Load and preprocess the dataset."""
    images, labels = [], []
    label_counts = {0: 0, 1: 0}

    for category in ["Tumor", "Normal"]:
        path = os.path.join(dataset_path, category)
        label = 1 if category == "Tumor" else 0
        image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:100]

        for img_name in image_files:
            img_path = os.path.join(path, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, (64, 64)) / 255.0
                images.append(image)
                labels.append(label)
                label_counts[label] += 1

    print(f"Loaded {label_counts[0]} Normal and {label_counts[1]} Tumor images.")
    return np.array(images).reshape(len(images), -1), np.array(labels)

def train_cnn_model(images, labels, model_save_path="models/cnn_model.h5"):
    """Train a CNN model and save it."""
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train = X_train.reshape(-1, 64, 64, 1)
    X_test = X_test.reshape(-1, 64, 64, 1)
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])

    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    _, cnn_accuracy = cnn_model.evaluate(X_test, y_test)
    print("CNN Accuracy:", cnn_accuracy)
    cnn_model.save(model_save_path)
