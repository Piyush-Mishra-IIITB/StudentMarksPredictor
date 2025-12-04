import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# --- Page Config ---
st.set_page_config(
    page_title="Student Score Predictor",
    layout="centered",
    page_icon="🎓"
)

# --- Title & Description ---
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Student Score Prediction 🎓</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict a student's performance based on personal info and scores.</p>", unsafe_allow_html=True)
st.write("---")

# --- Sidebar for Input ---
st.sidebar.header("Enter Student Details")
gender = st.sidebar.selectbox("Gender", ["male", "female"])
ethnicity = st.sidebar.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parental_education = st.sidebar.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.sidebar.selectbox("Lunch", ["standard", "free/reduced"])
prep_course = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"])
reading_score = st.sidebar.number_input("Reading Score", min_value=0, max_value=100, value=50)
writing_score = st.sidebar.number_input("Writing Score", min_value=0, max_value=100, value=50)

# --- Layout: Two Columns ---
col1, col2 = st.columns(2)
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910760.png", width=200)  # optional decorative image
with col2:
    st.markdown("### Enter details in the sidebar and click Predict!")

st.write("---")

# --- Prediction ---
if st.button("Predict"):
    # Prepare input
    data = CustomData(
        gender=gender,
        race_ethnicity=ethnicity,
        parental_level_of_education=parental_education,
        lunch=lunch,
        test_preparation_course=prep_course,
        reading_score=reading_score,
        writing_score=writing_score
    )

    input_df = data.get_data_as_data_frame()

    # Prediction
    pipeline = PredictPipeline()
    prediction = pipeline.predict(input_df)

    # --- Display Result ---
    st.success(f"🎯 Predicted Total Score: {prediction[0]:.2f}")

    # Optional: display input data in a table
    st.markdown("**Input Data:**")
    st.table(input_df)

    # Optional: visualize scores
    st.bar_chart(pd.DataFrame({
        "Scores": ["Reading", "Writing", "Predicted Total"],
        "Value": [reading_score, writing_score, prediction[0]]
    }).set_index("Scores"))
