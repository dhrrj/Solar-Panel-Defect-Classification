import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

MODEL_FILES = ["mobilenetv5.keras", "efficient_model.keras"]
IMG_HEIGHT = 180
IMG_WIDTH = 180

@st.cache_resource
def load_model():
    for model_name in MODEL_FILES:
        model_path = Path(model_name)
        if model_path.exists():
            return tf.keras.models.load_model(str(model_path))
    return None


def get_class_names(data_dir="Data"):
    data_path = Path(data_dir)
    if data_path.exists():
        return sorted([item.name for item in data_path.iterdir() if item.is_dir()])
    return [
        "Bird-drop",
        "Clean",
        "Dusty",
        "Electrical-damage",
        "Physical-Damage",
        "Snow-Covered",
    ]


def preprocess_image(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    array = np.array(image).astype("float32") / 255.0
    return np.expand_dims(array, axis=0)


def predict(image: Image.Image, model, class_names):
    image_array = preprocess_image(image)
    predictions = model.predict(image_array)
    index = int(np.argmax(predictions[0]))
    return class_names[index], predictions[0]


def main():
    st.set_page_config(page_title="Solar Panel Defect Classifier", layout="centered")
    st.title("Solar Panel Defect Classifier")
    st.write(
        "Upload a solar panel image to predict the defect category using the trained model."
    )

    model = load_model()
    class_names = get_class_names()

    if model is None:
        st.error(
            "No saved model found. Put `mobilenetv5.keras` or `efficient_model.keras` in the project root."
        )
        return

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                label, scores = predict(image, model, class_names)
                st.success(f"Predicted class: **{label}**")
                st.write("### Confidence scores")
                for name, score in zip(class_names, scores):
                    st.write(f"- {name}: {score:.2%}")


if __name__ == "__main__":
    main()
