import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")

st.title("🏦 Automated Loan Approval Predictor")
st.write("Enter applicant details to evaluate loan risk instantly.")

# Load the pre-trained .pkl file directly
@st.cache_resource
def load_model():
    return joblib.load('loan_model.pkl')

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Error loading model artifact 'loan_model.pkl': {e}")
    st.stop()

with st.form("loan_application_form"):
    st.subheader("1. Applicant Profile")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    with col2:
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    with col3:
        no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
    with col4:
        cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=700)

    st.subheader("2. Financial & Loan Request")
    col5, col6, col7 = st.columns(3)
    with col5:
        income_annum = st.number_input("Annual Income (₹)", min_value=100000, max_value=100000000, value=5000000, step=100000)
    with col6:
        loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=100000, max_value=100000000, value=15000000, step=100000)
    with col7:
        loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=30, value=10)

    st.subheader("3. Assets Evaluation")
    col8, col9, col10, col11 = st.columns(4)
    with col8:
        residential_assets_value = st.number_input("Residential Asset Value (₹)", min_value=0, max_value=100000000, value=4000000, step=100000)
    with col9:
        commercial_assets_value = st.number_input("Commercial Asset Value (₹)", min_value=0, max_value=100000000, value=2000000, step=100000)
    with col10:
        luxury_assets_value = st.number_input("Luxury Asset Value (₹)", min_value=0, max_value=100000000, value=10000000, step=100000)
    with col11:
        bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0, max_value=100000000, value=3000000, step=100000)

    submit = st.form_submit_button("Evaluate Application")

if submit:
    input_data = pd.DataFrame({
        'no_of_dependents': [no_of_dependents],
        'education': [education],
        'self_employed': [self_employed],
        'income_annum': [income_annum],
        'loan_amount': [loan_amount],
        'loan_term': [loan_term],
        'cibil_score': [cibil_score],
        'residential_assets_value': [residential_assets_value],
        'commercial_assets_value': [commercial_assets_value],
        'luxury_assets_value': [luxury_assets_value],
        'bank_asset_value': [bank_asset_value]
    })

    prediction = pipeline.predict(input_data)[0]
    probabilities = pipeline.predict_proba(input_data)[0]
    
    classes = list(pipeline.classes_)
    if 'Approved' in classes:
        approved_idx = classes.index('Approved')
    elif 1 in classes:
        approved_idx = classes.index(1)
    else:
        approved_idx = 1

    approval_prob = probabilities[approved_idx] * 100

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Approval Probability", f"{approval_prob:.1f}%")
    
    total_assets = residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value
    m2.metric("Total Asset Backing", f"₹{total_assets:,.0f}")

    if prediction in [1, 'Approved']:
        st.success(f"🎉 **Application Approved!** (Confidence: {approval_prob:.1f}%)")
    else:
        st.error(f"❌ **Application Rejected!** (Approval Probability: {approval_prob:.1f}%)")
