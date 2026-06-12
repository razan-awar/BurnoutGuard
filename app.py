import os
from datetime import datetime

import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb


# -----------------------------
# Load saved model files
# -----------------------------
model = joblib.load("models/burnout_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="BurnoutGuard",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 BurnoutGuard")
st.subheader("Student Burnout Risk Prediction System")
st.write("Fill in your weekly information to predict your burnout risk level.")


# -----------------------------
# User input form
# -----------------------------
st.markdown("### Personal Information")

age = st.number_input("Age", min_value=18, max_value=60, value=20)
gender = st.selectbox("Gender", list(label_encoders["gender"].classes_))
academic_year = st.selectbox("Academic Year", [1, 2, 3, 4])


st.markdown("### Academic Factors")

study_hours_per_day = st.slider("Study Hours Per Day", 0.0, 15.0, 4.0)
exam_pressure = st.slider("Exam Pressure", 1, 10, 5)
academic_performance = st.slider("Academic Performance", 1, 10, 7)


st.markdown("### Mental Health Factors")

stress_level = st.slider("Stress Level", 1, 10, 5)
anxiety_score = st.slider("Anxiety Score", 1, 10, 5)
depression_score = st.slider("Depression Score", 1, 10, 3)


st.markdown("### Lifestyle Factors")

sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
physical_activity = st.slider("Physical Activity Hours", 0.0, 10.0, 2.0)
screen_time = st.slider("Screen Time Hours", 0.0, 15.0, 5.0)


st.markdown("### Support & Environment")

social_support = st.slider("Social Support", 1, 10, 6)
financial_stress = st.slider("Financial Stress", 1, 10, 4)


# -----------------------------
# Hidden default values
# These are still needed because the model was trained with these columns
# -----------------------------
internet_usage = screen_time
family_expectation = 5
mental_health_index = 10 - ((stress_level + anxiety_score + depression_score) / 3)


# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict Burnout Risk"):

    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "academic_year": academic_year,
        "study_hours_per_day": study_hours_per_day,
        "exam_pressure": exam_pressure,
        "academic_performance": academic_performance,
        "stress_level": stress_level,
        "anxiety_score": anxiety_score,
        "depression_score": depression_score,
        "sleep_hours": sleep_hours,
        "physical_activity": physical_activity,
        "social_support": social_support,
        "screen_time": screen_time,
        "internet_usage": internet_usage,
        "financial_stress": financial_stress,
        "family_expectation": family_expectation,
        "mental_health_index": mental_health_index,

        # Feature Engineering
        "stress_sleep_ratio": stress_level / (sleep_hours + 1),
        "academic_burden": study_hours_per_day + exam_pressure,
        "wellbeing_score": sleep_hours + physical_activity + social_support - stress_level - financial_stress,
        "mental_pressure_score": stress_level + anxiety_score + depression_score + exam_pressure
    }])

    # Missing data handling
    input_data = input_data.fillna(0)

    # Encode categorical columns
    for column in input_data.columns:
        if column in label_encoders and column != "risk_level":
            input_data[column] = label_encoders[column].transform(
                input_data[column].astype(str)
            )

    # Same column order as training
    input_data = input_data[feature_columns]

    # -----------------------------
    # XGBoost prediction
    # -----------------------------
    prediction = model.predict(input_data)[0]
    risk_label = label_encoders["risk_level"].inverse_transform([prediction])[0]

    probabilities = model.predict_proba(input_data)[0]
    confidence = probabilities[prediction] * 100

    # -----------------------------
    # Display result
    # -----------------------------
    st.markdown("---")
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Risk", risk_label)

    with col2:
        st.metric("Model Confidence", f"{confidence:.2f}%")

    if risk_label == "High":
        st.error("🔴 High Burnout Risk")
        st.write(
            "Recommendation: Contact an academic advisor and focus on reducing stress, "
            "improving sleep, and balancing workload."
        )

    elif risk_label == "Medium":
        st.warning("🟡 Medium Burnout Risk")
        st.write(
            "Recommendation: Monitor stress, improve sleep habits, and manage academic pressure."
        )

    else:
        st.success("🟢 Low Burnout Risk")
        st.write(
            "Recommendation: Continue maintaining healthy habits and balanced study routines."
        )

    # -----------------------------
    # SHAP Explainability
    # -----------------------------
    st.markdown("---")
    st.subheader("Why did the model predict this?")

    try:
        dmatrix = xgb.DMatrix(input_data, feature_names=feature_columns)

        shap_values = model.get_booster().predict(
            dmatrix,
            pred_contribs=True
        )

        number_of_features = len(feature_columns)

        if len(shap_values.shape) == 3:
            impacts = shap_values[0, int(prediction), :-1]
        else:
            if shap_values.shape[1] == number_of_features + 1:
                impacts = shap_values[0, :-1]
            else:
                group_size = number_of_features + 1
                start = int(prediction) * group_size
                end = start + number_of_features
                impacts = shap_values[0, start:end]

        explanation_df = pd.DataFrame({
            "Feature": feature_columns,
            "SHAP Impact": impacts
        })

        explanation_df["Absolute Impact"] = explanation_df["SHAP Impact"].abs()

        explanation_df = explanation_df.sort_values(
            by="Absolute Impact",
            ascending=False
        ).head(5)

        st.write("Top factors that influenced the prediction:")

        for _, row in explanation_df.iterrows():
            feature = row["Feature"]
            impact = row["SHAP Impact"]

            if impact > 0:
                st.write(f"⬆️ **{feature}** increased the influence on the predicted risk.")
            else:
                st.write(f"⬇️ **{feature}** reduced the influence on the predicted risk.")

        with st.expander("View SHAP values table"):
            st.dataframe(
                explanation_df[["Feature", "SHAP Impact"]],
                use_container_width=True
            )

    except Exception as e:
        st.warning("SHAP explanation could not be displayed.")
        st.write("Error:", e)

    # -----------------------------
    # Personalized recommendations
    # -----------------------------
    st.markdown("---")
    st.subheader("Personalized Recommendations")

    recommendations = []

    if stress_level >= 7:
        recommendations.append("Reduce stress by planning study breaks and using relaxation techniques.")

    if sleep_hours < 6:
        recommendations.append("Try to improve sleep duration to at least 6–7 hours per night.")

    if anxiety_score >= 7:
        recommendations.append("Consider talking to an academic advisor or counselor.")

    if exam_pressure >= 7:
        recommendations.append("Create a weekly study plan to manage exam pressure.")

    if financial_stress >= 7:
        recommendations.append("Seek financial support options if available.")

    if social_support <= 4:
        recommendations.append("Try to increase social support through friends, family, or student services.")

    if len(recommendations) == 0:
        st.success("No major risk factors detected. Continue maintaining healthy habits.")
    else:
        for rec in recommendations:
            st.write("✅", rec)

    # -----------------------------
    # Save prediction history
    # -----------------------------
    history_row = pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "age": age,
        "gender": gender,
        "academic_year": academic_year,
        "risk_level": risk_label,
        "confidence": round(confidence, 2),
        "stress_level": stress_level,
        "sleep_hours": sleep_hours,
        "anxiety_score": anxiety_score,
        "exam_pressure": exam_pressure
    }])

    history_file = "predictions_history.csv"

    history_row.to_csv(
        history_file,
        mode="a",
        header=not os.path.exists(history_file),
        index=False
    )

    with st.expander("View Recent Prediction History"):
        if os.path.exists(history_file):
            history_df = pd.read_csv(history_file)
            st.dataframe(history_df.tail(10), use_container_width=True)
        else:
            st.info("No history available yet.")