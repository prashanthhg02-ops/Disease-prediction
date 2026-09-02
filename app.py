import streamlit as st

from disease_prediction import SYMPTOMS, predict_disease, train_model

st.set_page_config(page_title="Disease Prediction", page_icon="🩺", layout="wide")

@st.cache_resource
def get_model():
    model, accuracy = train_model()
    return model, accuracy

model, accuracy = get_model()

st.title("Disease Prediction System")
st.caption("AI-powered disease prediction based on selected symptoms")

st.write(f"Model accuracy on test data: {accuracy:.2f}")

selected_symptoms = []
with st.form("symptom_form"):
    cols = st.columns(4)
    for i, symptom in enumerate(SYMPTOMS):
        with cols[i % 4]:
            if st.checkbox(symptom, key=symptom):
                selected_symptoms.append(symptom)

    submitted = st.form_submit_button("Predict Disease")

if submitted:
    if not selected_symptoms:
        st.warning("Please select at least one symptom before predicting.")
    else:
        disease = predict_disease(selected_symptoms, model)
        st.success(f"Predicted disease: {disease}")
        st.write("Selected symptoms:", ", ".join(selected_symptoms))
