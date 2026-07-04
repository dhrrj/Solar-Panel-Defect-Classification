import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
IMG_HEIGHT = 180
IMG_WIDTH = 180
IMG_CHANNELS = 3
MODEL_CANDIDATES = [
    BASE_DIR / "mobilenetv5.keras",
    BASE_DIR / "efficient_model.keras",
    BASE_DIR / "solar_defect_model.keras",
    BASE_DIR / "best_model.keras",
]


@st.cache_resource(show_spinner=False)
def load_model(model_path: str | None = None):
    if model_path and os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        class_names = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])
        return model, class_names

    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            model = tf.keras.models.load_model(candidate)
            class_names = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])
            return model, class_names

    raise FileNotFoundError("No trained model was found. Train one first or place a .keras model in the project folder.")


@st.cache_data(show_spinner=False)
def get_class_names():
    return sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])


@st.cache_resource(show_spinner=False)
def train_fallback_model():
    class_names = get_class_names()
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(len(class_names), activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=3)
    output_path = BASE_DIR / "solar_defect_model.keras"
    model.save(output_path)
    return model, class_names


def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = ImageOps.fit(image, (IMG_WIDTH, IMG_HEIGHT), method=Image.Resampling.LANCZOS)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image, image_array


st.set_page_config(page_title="Solar Panel Defect Classifier", layout="centered")
st.title("Solar Panel Defect Classifier")
st.write("Upload a solar panel image to predict the defect category.")

with st.sidebar:
    st.header("Model options")
    st.write("Place your trained .keras model in the project folder or train one here.")
    chosen_model_path = st.text_input(
        "Model path",
        value=str(next((p for p in MODEL_CANDIDATES if p.exists()), "")),
        help="Example: C:/Users/Lenovo/OneDrive/Desktop/Solar Panel Defect/mobilenetv5.keras",
    )
    if st.button("Train a new model from the Data folder"):
        with st.spinner("Training model from your dataset. This may take a few minutes..."):
            model, class_names = train_fallback_model()
            st.session_state["model"] = model
            st.session_state["class_names"] = class_names
            st.success("Training complete. The model was saved as solar_defect_model.keras")

if "model" not in st.session_state:
    st.session_state["model"] = None
if "class_names" not in st.session_state:
    st.session_state["class_names"] = get_class_names()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image, image_array = preprocess_image(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.session_state["model"] is None:
        try:
            st.session_state["model"], st.session_state["class_names"] = load_model(chosen_model_path or None)
        except FileNotFoundError:
            st.warning("No trained model was found. Use the sidebar button to train a model from the Data folder.")
            st.stop()

    with st.spinner("Predicting defect type..."):
        prediction = st.session_state["model"].predict(image_array, verbose=0)[0]
        top_idx = np.argsort(prediction)[::-1][:3]

    st.subheader("Prediction")
    for idx in top_idx:
        label = st.session_state["class_names"][idx]
        prob = float(prediction[idx] * 100)
        st.write(f"{label.replace('-', ' ')}: {prob:.2f}%")

    top_class = st.session_state["class_names"][int(np.argmax(prediction))]
    st.success(f"Predicted class: {top_class.replace('-', ' ')}")
else:
    st.info("Upload an image to begin classification.")
