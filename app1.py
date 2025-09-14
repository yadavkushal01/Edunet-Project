import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = (128, 128)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Page config
st.set_page_config(page_title="♻️ Waste Classification", layout="centered")

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
st.markdown('<div class="main-title">♻️ Waste Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload an image of waste material and let Model Classify It!</div>', unsafe_allow_html=True)

# --- Upload Section ---
uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg", "jpeg", "png"], help="Upload a waste material image")

if uploaded_file is not None and model is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="🖼 Image Uploaded", use_container_width=True)

    # Process image
    img = load_img(uploaded_file, target_size=IMG_SIZE)
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Prediction
    pred = model.predict(x)
    pred_class = np.argmax(pred)
    confidence = np.max(pred)

    # Display result in styled box
    mess=""
    if CLASS_NAMES[pred_class] not in ['trash']:
        mess="Recyclable"
    else:
        mess="Non-Recyclable"

    st.markdown(
        f"""
        <div class="result-box">
            <h3 style= "color: #006666;">🔍 Prediction: <b>{mess}</b> <b>{CLASS_NAMES[pred_class].capitalize()}</b></h3>
            <p style="font-size:18px; color: #006666;">Confidence: <b>{confidence * 100:.2f}%</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Confidence distribution chart
    st.subheader("📊 Probabilities")

    probs_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability": pred[0]
    })

    

    explode = [0.1 if i == pred_class else 0 for i in range(len(CLASS_NAMES))]

    labels = [
        CLASS_NAMES[i] if pred[0][i] > 0.01 else ""  # show only >1%
        for i in range(len(CLASS_NAMES))
    ]

    fig, ax = plt.subplots()
    ax.pie(
        probs_df["Probability"],
        labels=labels,
        autopct=lambda p: f"{p:.1f}%" if p > 1 else "",  # hide <1%
        startangle=90,
        explode=explode,
        colors=plt.cm.Set3.colors,  # nice color palette
        textprops={"fontsize": 12}
    )
    ax.axis("equal")  # Circle shape
    st.pyplot(fig)

else:
    st.info("⬆️ Please upload an image above to start classification.")
