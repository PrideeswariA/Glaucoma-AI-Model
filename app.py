import os
import io
import csv
import time
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
import matplotlib.pyplot as plt


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="OptiVision AI | Glaucoma Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #10233f;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #142b4a;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #64748b;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.04);
    }

    .metric-label {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #102a43;
        font-size: 28px;
        font-weight: 800;
        margin-top: 7px;
    }

    .section-title {
        color: #142b4a;
        font-size: 23px;
        font-weight: 750;
        margin-top: 12px;
    }

    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 25px 0 10px 0;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. CONSTANTS
# =========================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mobilenetv2_glaucoma_model.keras"
)

IMAGE_SIZE = (224, 224)

# Original model class indices:
# glaucoma = 0
# normal = 1
#
# Sigmoid output corresponds to class index 1 (Normal).
# Therefore, glaucoma score = 1 - model output.

TEST_ACCURACY = 90.95
TEST_LOSS = 0.2140

GLAUCOMA_PRECISION = 0.8807
GLAUCOMA_RECALL = 0.9412
GLAUCOMA_F1 = 0.9100

NORMAL_PRECISION = 0.9406
NORMAL_RECALL = 0.8796
NORMAL_F1 = 0.9091

CONFUSION_MATRIX = np.array([
    [96, 6],
    [13, 95]
])

CLASS_NAMES = ["Glaucoma", "Normal"]


# =========================================================
# 4. SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# =========================================================
# 5. LOAD MODEL
# =========================================================

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# =========================================================
# 6. IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):
    """
    Convert image into the format expected by the model.

    The training pipeline used:
        ImageDataGenerator(rescale=1./255)

    The model contains a Rescaling layer with:
        scale=2, offset=-1

    Therefore, this function only divides pixels by 255.
    """

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =========================================================
# 7. MAKE PREDICTION
# =========================================================

def predict_image(model, image):
    processed_image = preprocess_image(image)

    output = model.predict(
        processed_image,
        verbose=0
    )

    output = np.asarray(output).reshape(-1)

    if len(output) != 1:
        raise ValueError(
            "Unexpected model output. "
            "This app expects a binary sigmoid model."
        )

    normal_score = float(output[0])

    # Keep scores within valid probability bounds.
    normal_score = float(
        np.clip(normal_score, 0.0, 1.0)
    )

    glaucoma_score = 1.0 - normal_score

    if glaucoma_score >= normal_score:
        prediction = "Glaucoma"
        confidence = glaucoma_score
    else:
        prediction = "Normal"
        confidence = normal_score

    return {
        "prediction": prediction,
        "glaucoma_score": glaucoma_score,
        "normal_score": normal_score,
        "confidence": confidence,
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }


# =========================================================
# 8. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown("# 👁️ OptiVision AI")

    st.caption("AI-Powered Eye Image Screening")

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "AI Screening",
            "Prediction History",
            "Model Performance",
            "About Project"
        ],
        key="navigation"
    )

    st.markdown("---")

    st.markdown("### 🧠 AI Model")

    st.write("MobileNetV2")

    st.write("Binary Classification")

    st.write("Input: 224 × 224 RGB")

    st.markdown("---")

    st.caption("Academic / Educational Project")


# =========================================================
# 9. PAGE HEADER
# =========================================================

st.markdown(
    '<div class="main-title">OptiVision AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Glaucoma Screening Using Deep Learning'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# 10. LOAD MODEL STATUS
# =========================================================

model = load_model()

if model is None:
    st.error(
        "Model file not found. Please ensure that "
        "'mobilenetv2_glaucoma_model.keras' is present "
        "in the same directory as app.py."
    )

    st.info(
        "Upload the trained model file to the root of "
        "your GitHub repository."
    )


# =========================================================
# 11. DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown("## 📊 Project Dashboard")

    st.write(
        "Welcome to OptiVision AI. This application "
        "demonstrates binary classification of retinal "
        "fundus images into Glaucoma and Normal categories."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Test Accuracy</div>
                <div class="metric-value">{TEST_ACCURACY:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">AI Architecture</div>
                <div class="metric-value">MobileNetV2</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Classification</div>
                <div class="metric-value">Binary</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">How It Works</div>',
        unsafe_allow_html=True
    )

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown(
            """
            <div class="info-card">
                <h3>📤 1. Upload</h3>
                Upload a retinal fundus image in JPG, JPEG,
                or PNG format.
            </div>
            """,
            unsafe_allow_html=True
        )

    with step2:
        st.markdown(
            """
            <div class="info-card">
                <h3>🧠 2. Analyze</h3>
                The trained MobileNetV2 model analyzes
                the uploaded image.
            </div>
            """,
            unsafe_allow_html=True
        )

    with step3:
        st.markdown(
            """
            <div class="info-card">
                <h3>📋 3. View Result</h3>
                View the predicted class and model scores.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("### 📁 Dataset Summary")

    dataset_data = pd.DataFrame({
        "Dataset Split": ["Training", "Validation", "Testing"],
        "Number of Images": [1455, 416, 210]
    })

    st.dataframe(
        dataset_data,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The model was trained using Glaucoma and Normal "
        "retinal images. The original dataset also contains "
        "other categories, which are not classified by "
        "this binary model."
    )

    if st.button("Open AI Screening", key="dashboard_screening"):
        st.session_state.navigation = "AI Screening"
        st.rerun()


# =========================================================
# 12. AI SCREENING
# =========================================================

elif page == "AI Screening":

    st.markdown("## 🔬 AI Image Screening")

    st.write(
        "Upload a retinal fundus image to obtain a model "
        "prediction. Use an appropriate retinal image "
        "for meaningful results."
    )

    uploaded_file = st.file_uploader(
        "Choose a retinal fundus image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            left, right = st.columns([1, 1])

            with left:
                st.markdown("### 🖼️ Uploaded Image")

                st.image(
                    image,
                    caption="Uploaded retinal fundus image",
                    use_container_width=True
                )

                st.caption(
                    f"Image dimensions: "
                    f"{image.width} × {image.height}"
                )

            with right:

                st.markdown("### 🧠 Analysis")

                if model is None:

                    st.error(
                        "Prediction is unavailable because "
                        "the trained model could not be loaded."
                    )

                else:

                    if st.button(
                        "🚀 Analyze Image",
                        type="primary",
                        key="analyze_image"
                    ):

                        try:

                            with st.spinner(
                                "Analyzing the uploaded image..."
                            ):

                                result = predict_image(
                                    model,
                                    image
                                )

                            st.session_state.prediction_result = result

                            # Add each analysis to session history.
                            history_record = {
                                "Timestamp": result["timestamp"],
                                "Prediction": result["prediction"],
                                "Glaucoma Score (%)": round(
                                    result["glaucoma_score"] * 100,
                                    2
                                ),
                                "Normal Score (%)": round(
                                    result["normal_score"] * 100,
                                    2
                                )
                            }

                            st.session_state.history.append(
                                history_record
                            )

                        except Exception as error:

                            st.error(
                                "Unable to analyze this image."
                            )

                            st.exception(error)

            result = st.session_state.prediction_result

            if result is not None:

                st.markdown("---")

                st.markdown("## 📋 Prediction Result")

                result_col1, result_col2, result_col3 = st.columns(3)

                with result_col1:

                    st.metric(
                        "Predicted Class",
                        result["prediction"]
                    )

                with result_col2:

                    st.metric(
                        "Glaucoma Score",
                        f"{result['glaucoma_score'] * 100:.2f}%"
                    )

                with result_col3:

                    st.metric(
                        "Normal Score",
                        f"{result['normal_score'] * 100:.2f}%"
                    )

                st.markdown("### Model Output Scores")

                score_df = pd.DataFrame({
                    "Class": ["Glaucoma", "Normal"],
                    "Model Score (%)": [
                        result["glaucoma_score"] * 100,
                        result["normal_score"] * 100
                    ]
                })

                st.bar_chart(
                    score_df.set_index("Class")
                )

                st.caption(
                    "These scores are derived from the model output "
                    "and are not clinically calibrated probabilities."
                )

                st.warning(
                    "This is an educational AI demonstration, not a "
                    "medical diagnosis. A model prediction cannot "
                    "confirm or rule out glaucoma. Consult a qualified "
                    "eye-care professional for examination and diagnosis."
                )

        except Exception:

            st.error(
                "The uploaded file could not be read as an image. "
                "Please upload a valid JPG, JPEG, or PNG file."
            )


# =========================================================
# 13. PREDICTION HISTORY
# =========================================================

elif page == "Prediction History":

    st.markdown("## 🗂️ Prediction History")

    st.write(
        "This page displays predictions made during the "
        "current app session."
    )

    if len(st.session_state.history) == 0:

        st.info(
            "No predictions available yet. Go to AI Screening "
            "and analyze an image first."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download History as CSV",
            data=csv_data,
            file_name="glaucoma_prediction_history.csv",
            mime="text/csv"
        )

        if st.button(
            "🗑️ Clear Prediction History",
            key="clear_history"
        ):

            st.session_state.history = []
            st.session_state.prediction_result = None

            st.success("Prediction history cleared.")

            st.rerun()


# =========================================================
# 14. MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.markdown("## 📈 Model Performance")

    st.write(
        "The following metrics represent the previously "
        "recorded evaluation of this model on the test dataset."
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Test Accuracy</div>
                <div class="metric-value">{TEST_ACCURACY:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Test Loss</div>
                <div class="metric-value">{TEST_LOSS:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric3:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Model</div>
                <div class="metric-value">MobileNetV2</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("### Confusion Matrix")

    st.caption(
        "Rows represent actual classes; columns represent "
        "predicted classes. Class order: Glaucoma, Normal."
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    image_plot = ax.imshow(
        CONFUSION_MATRIX,
        interpolation="nearest"
    )

    ax.set_title("Test Dataset Confusion Matrix")

    ax.set_xlabel("Predicted Class")

    ax.set_ylabel("Actual Class")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(CLASS_NAMES)

    for row in range(CONFUSION_MATRIX.shape[0]):

        for col in range(CONFUSION_MATRIX.shape[1]):

            ax.text(
                col,
                row,
                str(CONFUSION_MATRIX[row, col]),
                ha="center",
                va="center",
                fontsize=13
            )

    fig.colorbar(
        image_plot,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.markdown("### Classification Report")

    report_df = pd.DataFrame({
        "Class": ["Glaucoma", "Normal"],
        "Precision": [
            GLAUCOMA_PRECISION,
            NORMAL_PRECISION
        ],
        "Recall": [
            GLAUCOMA_RECALL,
            NORMAL_RECALL
        ],
        "F1-Score": [
            GLAUCOMA_F1,
            NORMAL_F1
        ],
        "Support": [102, 108]
    })

    st.dataframe(
        report_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Evaluation metrics are based on the recorded test results. "
        "They do not establish clinical effectiveness or suitability "
        "for medical decision-making."
    )


# =========================================================
# 15. ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.markdown("## ℹ️ About the Project")

    st.write(
        "OptiVision AI is an academic deep-learning project "
        "that demonstrates binary classification of retinal "
        "fundus images into Glaucoma and Normal categories."
    )

    st.markdown("### 🎯 Project Objectives")

    st.markdown(
        """
        - Demonstrate transfer learning for retinal image classification.
        - Use MobileNetV2 as the feature extraction backbone.
        - Present model predictions through a Streamlit web application.
        - Display evaluation metrics and a confusion matrix.
        - Maintain a session-based history of image predictions.
        """
    )

    st.markdown("### 🧠 Model Architecture")

    st.code(
        """
Input Image (224 × 224 × 3)
          |
          v
Image Preprocessing
          |
          v
MobileNetV2
(Pretrained ImageNet Weights)
          |
          v
Global Average Pooling
          |
          v
Dropout (0.3)
          |
          v
Dense Layer (1, Sigmoid)
          |
          v
Glaucoma / Normal Score
        """,
        language="text"
    )

    st.markdown("### 📚 Dataset")

    st.write(
        "The project uses the Cataract, DR, Normal and "
        "Glaucoma Fundus Images dataset available on Kaggle."
    )

    st.markdown(
        """
        Dataset:
        https://www.kaggle.com/datasets/drskprabhakar/cataract-dr-normal-glaucoma-fundus-images-dataset
        """
    )

    st.markdown("### 🛠️ Technologies Used")

    st.markdown(
        """
        - Python
        - TensorFlow / Keras
        - MobileNetV2
        - Streamlit
        - NumPy and Pandas
        - Matplotlib
        - Pillow
        """
    )

    st.markdown("### ⚠️ Important Disclaimer")

    st.warning(
        "This application is intended only for academic and "
        "educational demonstration. It is not a clinically validated "
        "medical device and must not be used to diagnose, treat, "
        "or rule out glaucoma. Medical decisions must be made with "
        "qualified healthcare professionals."
    )


# =========================================================
# 16. FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        OptiVision AI | Glaucoma Screening Project<br>
        Built with Python, TensorFlow and Streamlit<br>
        For educational purposes only — not for clinical diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)
