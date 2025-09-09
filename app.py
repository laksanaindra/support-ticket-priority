import streamlit as st
import pandas as pd
import joblib

model = joblib.load("xgb_model.pkl")

st.title("🎫 Support Ticket Priority Prediction")

customers_affected = st.number_input("Customers Affected", min_value=0)
downtime_min = st.number_input("Downtime (minutes)", min_value=0)
error_rate_pct = st.number_input("Error Rate (%)", min_value=0.0, max_value=100.0)

company_size = st.selectbox("Company Size", [1,2,3])
customer_tier = st.selectbox("Customer Tier", [1,2,3])
region = st.selectbox("Region", [1,2,3])
product_area = st.selectbox("Product Area", [1,2,3,4,5,6])


input_data = pd.DataFrame({
    "customers_affected": [customers_affected],
    "downtime_min": [downtime_min],
    "error_rate_pct": [error_rate_pct],
    "company_size_cat": [company_size],
    "customer_tier_cat": [customer_tier],
    "region_cat": [region],
    "product_area_cat": [product_area],
   
})

if st.button("Predict"):
    pred = model.predict(input_data)[0]
    mapping = {0:"Low", 1:"Medium", 2:"High"}
    st.success(f"Predicted Priority: **{mapping[pred]}**")
