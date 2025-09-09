import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("xgb_model.pkl")

# List kolom sesuai training
feature_names = [
    'company_size_cat', 'industry_cat', 'customer_tier_cat', 'org_users', 'region_cat',
    'past_30d_tickets', 'past_90d_incidents', 'product_area_cat', 'booking_channel_cat',
    'reported_by_role_cat', 'customers_affected', 'error_rate_pct', 'downtime_min',
    'payment_impact_flag', 'security_incident_flag', 'data_loss_flag',
    'has_runbook', 'customer_sentiment_cat'
]

st.title("🎫 Support Ticket Priority Prediction")

# === Input User (hanya fitur penting) ===
customers_affected = st.number_input("Customers Affected", min_value=0)
downtime_min = st.number_input("Downtime (minutes)", min_value=0)
error_rate_pct = st.number_input("Error Rate (%)", min_value=0.0, max_value=100.0)

company_size = st.selectbox("Company Size", [1,2,3])
customer_tier = st.selectbox("Customer Tier", [1,2,3])
customer_sentiment = st.selectbox("Customer Sentiment", [1,2,3])
region = st.selectbox("Region", [1,2,3])
product_area = st.selectbox("Product Area", [1,2,3,4,5,6])

# === Buat dataframe input dengan default values ===
# Default semua kolom = 0
input_dict = {col: 0 for col in feature_names}

# Isi kolom yang user input
input_dict.update({
    "customers_affected": customers_affected,
    "downtime_min": downtime_min,
    "error_rate_pct": error_rate_pct,
    "company_size_cat": company_size,
    "customer_tier_cat": customer_tier,
    "customer_sentiment_cat": customer_sentiment,
    "region_cat": region,
    "product_area_cat": product_area
})

# Buat dataframe 1 baris
input_data = pd.DataFrame([input_dict])

# === Prediksi ===
if st.button("Predict"):
    pred = model.predict(input_data)[0]
    mapping = {0:"Low", 1:"Medium", 2:"High"}
    st.success(f"Predicted Priority: **{mapping[pred]}**")
