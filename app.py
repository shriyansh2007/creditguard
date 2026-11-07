
import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(
    page_title="CreditGuard",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = "creditguard_artifacts/creditguard_model.pkl"
METADATA_PATH = "creditguard_artifacts/metadata.json"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r") as f:
        return json.load(f)

model = load_model()
metadata = load_metadata()

st.title("🛡️ CreditGuard")
st.subheader("Intelligent Credit Risk Assessment System")

st.write(
    "Assess historical good/bad credit risk using a machine learning model."
)

st.info(
    "This educational prototype predicts the historical credit label. "
    "It is not a calibrated probability of future loan default and "
    "should not be used as the sole basis for lending decisions."
)

st.write("Selected model:", metadata["model_name"])

st.divider()

st.header("Applicant Information")

features = metadata["features"]
categorical = metadata["categorical_features"]
numerical = metadata["numerical_features"]

applicant = {}

with st.form("applicant_form"):

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):

        container = col1 if i % 2 == 0 else col2

        with container:

            if feature in categorical:

                options = metadata["category_options"][feature]

                applicant[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    options=options
                )

            else:

                default = metadata["numeric_defaults"][feature]
                minimum = metadata["numeric_min"][feature]
                maximum = metadata["numeric_max"][feature]

                applicant[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=float(default),
                    step=1.0
                )

    submitted = st.form_submit_button(
        "Assess Credit Risk",
        use_container_width=True
    )

if submitted:

    input_df = pd.DataFrame(
        [applicant],
        columns=features
    )

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    bad_index = list(model.classes_).index(2)

    bad_score = probabilities[bad_index]

    st.divider()
    st.header("Assessment Result")

    col1, col2 = st.columns(2)

    with col1:

        if prediction == 1:
            st.success("Predicted category: Good Credit")
        else:
            st.error("Predicted category: Bad Credit")

    with col2:

        st.metric(
            "Model probability of bad-credit label",
            f"{bad_score * 100:.2f}%"
        )

    st.progress(float(bad_score))

    st.caption(
        "This is the model's estimated probability of the historical "
        "bad-credit class, not a calibrated probability of future default."
    )

    st.subheader("Applicant Details")
    st.dataframe(input_df, use_container_width=True)

st.divider()

st.caption(
    "CreditGuard | Educational Credit Risk Analytics Project"
)
