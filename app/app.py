import streamlit as st
import pickle
import os
from helper import transform_text

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load vectorizer and model
tfidf = pickle.load(
    open(
        os.path.join(BASE_DIR, "models", "vectorizer.pkl"),
        "rb"
    )
)

model = pickle.load(
    open(
        os.path.join(BASE_DIR, "models", "model.pkl"),
        "rb"
    )
)

# Streamlit UI
st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):

    # preprocess
    transformed_sms = transform_text(input_sms)

    # vectorize
    vector_input = tfidf.transform([transformed_sms])

    # predict
    result = model.predict(vector_input)[0]

    # display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")