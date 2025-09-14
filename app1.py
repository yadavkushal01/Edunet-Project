import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import altair as alt

# Constants
IMG_SIZE = (128, 128)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Page config
st.set_page_config(page_title="♻️ Smart Garbage Classification", layout="centered")

# Load trained model
@st.cache_resource
def load_garbage_model():
    try:
        model = load_model("garbage_classifier.h5")
        return model
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
        return None

model = load_garbage_model()

# --- Custom CSS ---
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            color: #2E8B57;
            font-size: 42px;
            font-weight: bold;
        }
        .sub-title {
            text-align: center;
            color: gray;
            font-size: 18px;
            margin-bottom: 30px;
        }
        .result-box {
            padding: 20px;
            border-radius: 12px;
            background-color: #f4f9f4;
            border: 1px solid #d4e6d4;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Page Title ---
st.markdown('<div class="main-title">♻️ Smart Garbage Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload an image of waste material and let AI classify it!</div>', unsafe_allow_html=True)

# --- Upload Section ---
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"], help="Upload a waste material image")

if uploaded_file is not None and model is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="🖼 Uploaded Image", use_container_width=True)

    # Process image
    img = load_img(uploaded_file, target_size=IMG_SIZE)
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Prediction
    pred = model.predict(x)
    pred_class = np.argmax(pred)
    confidence = np.max(pred)

    # Display result in styled box
    st.markdown(
        f"""
        <div class="result-box">
            <h3 style= "color: #006666;">🔍 Prediction: <b>{CLASS_NAMES[pred_class].capitalize()}</b></h3>
            <p style="font-size:18px; color: #006666;">Confidence: <b>{confidence * 100:.2f}%</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Confidence distribution chart
    st.subheader("📊 Class Probabilities")

    probs_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability": pred[0]
    })

    # Horizontal bar chart using Altair
    chart = (
        alt.Chart(probs_df)
        .mark_bar(cornerRadius=5)
        .encode(
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Class:N", sort="-x"),
            color=alt.Color("Class:N", legend=None, scale=alt.Scale(scheme="set2")),
            tooltip=["Class", alt.Tooltip("Probability", format=".2%")]
        )
        .properties(height=300)
    )

    st.altair_chart(chart, use_container_width=True)

else:
    st.info("⬆️ Please upload an image above to start classification.")
