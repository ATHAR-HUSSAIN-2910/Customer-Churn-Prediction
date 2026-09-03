import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load model
model_path = Path(__file__).parent.parent / "models" / "pipeline.pkl"

if not model_path.exists():
    st.error(f"❌ Model not found at: {model_path}")
    st.stop()

model = joblib.load(model_path)

# Page config
st.set_page_config(page_title="Churn Predictor", layout="wide")

# Title
st.title("📊 Customer Churn Prediction")

# ✅ MOBILE HINT - Shows on all devices
st.markdown("""
<style>
    .sidebar-hint {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 14px 20px;
        border-radius: 12px;
        margin: 12px 0 20px 0;
        text-align: center;
        font-size: 15px;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        animation: pulse 2s infinite;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .sidebar-hint strong {
        background: rgba(255,255,255,0.2);
        padding: 2px 12px;
        border-radius: 6px;
        font-size: 20px;
        margin: 0 4px;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.85; transform: scale(1.01); }
    }
    /* On mobile, make it more compact */
    @media screen and (max-width: 768px) {
        .sidebar-hint {
            font-size: 14px;
            padding: 12px 16px;
        }
        .sidebar-hint strong {
            font-size: 18px;
        }
    }
</style>

<div class="sidebar-hint">
    👆 Tap <strong>>></strong> in the top-left corner to open input panel
</div>
""", unsafe_allow_html=True)

st.markdown("Enter customer details to predict churn risk.")

# Sidebar inputs
with st.sidebar:
    st.header("👤 Customer Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

    st.header("💰 Account Information")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 50.0, step=5.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", 
                                  ["Electronic check", "Mailed check", 
                                   "Bank transfer (automatic)", "Credit card (automatic)"])

    st.header("📡 Services")
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# Build input dataframe
def build_input():
    services = [phone_service, internet_service, online_security, tech_support, 
                streaming_tv, streaming_movies]
    num_services = sum(1 for s in services if s != "No" and s != "No internet service")
    
    if tenure <= 12:
        tenure_group = "New"
    elif tenure <= 24:
        tenure_group = "Short-term"
    elif tenure <= 48:
        tenure_group = "Medium-term"
    else:
        tenure_group = "Long-term"
    
    return pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': ["No" if phone_service == "No" else "Yes"],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': ["No" if internet_service == "No" else "No"],
        'DeviceProtection': ["No" if internet_service == "No" else "No"],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [tenure * monthly_charges],
        'NumberOfServices': [num_services],
        'TenureGroup': [tenure_group]
    })

# Predict button
if st.button("🔮 Predict Churn", type="primary"):
    with st.spinner("Analyzing customer data..."):
        input_df = build_input()
        prediction = model.predict(input_df)[0]
        probability = float(model.predict_proba(input_df)[0][1])

    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error(f"⚠️ **High Churn Risk** - {probability:.1%} probability")
            st.warning("Consider offering retention incentives!")
        else:
            st.success(f"✅ **Low Churn Risk** - {(1-probability):.1%} probability")
            st.info("Customer is likely to stay.")

    with col2:
        st.metric("Churn Probability", f"{probability:.1%}")
        
        if probability < 0.3:
            st.progress(probability, text=f"🟢 Low Risk - {probability:.1%}")
        elif probability < 0.6:
            st.progress(probability, text=f"🟡 Medium Risk - {probability:.1%}")
        else:
            st.progress(probability, text=f"🔴 High Risk - {probability:.1%}")