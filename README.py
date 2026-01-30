## deploy
##use deploy to create a form for client
import streamlit as st
import pickle
from PIL import Image
import os

@st.cache_resource
def load_model_and_scaler():
    model_path = "model_knn.sav"
    scaler_path = "scaler_knn.save"

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None

    if not os.path.exists(scaler_path):
        st.error(f"Scaler file not found: {scaler_path}")
        return None, None

    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    return model, scaler


def main():
    st.set_page_config(page_title="Heart Failure Prediction", layout="centered")
    st.title(":red[HEART FAILURE PREDICTION]")

    # Image (optional)
    if os.path.exists("heart.jpg"):
        image = Image.open("heart.jpg")
        st.image(image, width=600)
    else:
        st.info("Add a file named 'heart.jpg' in the same folder to display an image.")

    st.markdown("### Enter patient details:")

    # Inputs
    col1, col2 = st.columns(2)

    with col1:
        age = st.text_input("Age", "")
        cp = st.text_input("Chest pain type (cp)", "")
        trestbps = st.text_input("Resting blood pressure (trestbps)", "")
        chol = st.text_input("Cholesterol (chol)", "")
        fbs = st.text_input("Fasting blood sugar (fbs)", "")
        restecg = st.text_input("Resting ECG (restecg)", "")

    with col2:
        sex = st.radio("Sex", ["Male", "Female"])
        thalach = st.text_input("Max heart rate (thalach)", "")
        exang = st.text_input("Exercise induced angina (exang)", "")
        oldpeak = st.text_input("ST depression (oldpeak)", "")
        slope = st.text_input("Slope of ST segment (slope)", "")
        ca = st.text_input("Number of major vessels (ca)", "")
        thal = st.text_input("Thalassemia (thal)", "")

    sex_val = 1 if sex == "Male" else 0

    model, scaler = load_model_and_scaler()

    if st.button("PREDICT"):
        if model is None or scaler is None:
            st.error("Model or scaler not loaded. Check files in the project folder.")
            return

        try:
            features = [
                float(age),
                sex_val,
                float(cp),
                float(trestbps),
                float(chol),
                float(fbs),
                float(restecg),
                float(thalach),
                float(exang),
                float(oldpeak),
                float(slope),
                float(ca),
                float(thal),
            ]

        except ValueError:
            st.error("Please enter valid numeric values for all fields.")
            return

        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)

        if prediction[0] == 0:
            st.success("Prediction: Not suffering from heart disease.")
            st.balloons()
        else:
            st.error("Prediction: Suffering from heart disease.")


if __name__ == "__main__":
    main()

