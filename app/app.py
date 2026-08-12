import os
import pickle
import streamlit as st

from helper import transform_text


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")


# --------------------------------------------------
# Load Model & Vectorizer
# --------------------------------------------------

@st.cache_resource
def load_models():
    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return vectorizer, model


try:
    tfidf, model = load_models()
except Exception as e:
    st.error("Unable to load the model. Please check the model files.")
    st.stop()


# --------------------------------------------------
# Custom Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .result {
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
    }

    .spam {
        background-color: #ffe5e5;
        color: #c62828;
    }

    .safe {
        background-color: #e5f7e9;
        color: #2e7d32;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📩 SMS Spam Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Detect whether an SMS message is Spam or Not Spam using Machine Learning</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Input
# --------------------------------------------------

input_sms = st.text_area(
    "Enter your SMS message",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=150
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict Message", use_container_width=True):

    if not input_sms.strip():
        st.warning("Please enter an SMS message first.")

    else:
        with st.spinner("Analyzing message..."):

            # Preprocess
            transformed_sms = transform_text(input_sms)

            # Vectorize
            vector_input = tfidf.transform([transformed_sms])

            # Prediction
            result = model.predict(vector_input)[0]

        st.divider()

        if result == 1:
            st.markdown(
                """
                <div class="result spam">
                    🚨 Spam Message
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "This message appears to contain characteristics commonly associated with spam."
            )

        else:
            st.markdown(
                """
                <div class="result safe">
                    ✅ Not Spam
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "This message appears to be a legitimate SMS."
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Machine Learning SMS Classification • Built with Python, Scikit-learn & Streamlit"
)