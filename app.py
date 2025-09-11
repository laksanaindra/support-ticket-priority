import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

# === Judul & Deskripsi ===
st.title("🎫 Support Ticket Priority Prediction")
st.markdown("""
Aplikasi ini memprediksi **prioritas tiket (Low, Medium, High)** berdasarkan informasi awal tiket.  
Model yang digunakan adalah **XGBoost**, hasil dari final project Data Science.  
""")

# === Input User ===
st.subheader("Masukkan Data Ticket")

customers_affected = st.number_input("Customers Affected", min_value=0)
downtime_min = st.number_input("Downtime (minutes)", min_value=0)
error_rate_pct = st.number_input("Error Rate (%)", min_value=0.0, max_value=100.0)

company_size = st.selectbox("Company Size (1=Small, 2=Medium, 3=Large)", [1,2,3])
customer_tier = st.selectbox("Customer Tier (1=Basic, 2=Plus, 3=Enterprise)", [1,2,3])
customer_sentiment = st.selectbox("Customer Sentiment (1=Negatif, 2=Netral, 3=Positif)", [1,2,3])
region = st.selectbox("Region (1=AMER, 2=EMEA, 3=APAC)", [1,2,3])
product_area = st.selectbox("Product Area (1=Auth, 2=Billing, 3=Mobile, 4=Data Pipeline, 5=Analytic, 6=Notifications)", [1,2,3,4,5,6])

reported_by_role = st.selectbox("Reported By Role (1=Support, 2=Devops, 3=Product Manager, 4=Finance, 5=C Level)", [1,2,3,4,5])
payment_impact_flag = st.selectbox("Payment Impact Flag (No=0, Yes=1)", [0,1])  # 0=No, 1=Yes

# === Buat dataframe input dengan default values ===
input_dict = {col: 0 for col in feature_names}
input_dict.update({
    "customers_affected": customers_affected,
    "downtime_min": downtime_min,
    "error_rate_pct": error_rate_pct,
    "company_size_cat": company_size,
    "customer_tier_cat": customer_tier,
    "customer_sentiment_cat": customer_sentiment,
    "region_cat": region,
    "product_area_cat": product_area,
    "reported_by_role_cat": reported_by_role,
    "payment_impact_flag": payment_impact_flag
})
input_data = pd.DataFrame([input_dict])

# === Prediksi ===
if st.button("Predict"):
    pred = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]  # probabilitas prediksi
    
    mapping = {0:"Low", 1:"Medium", 2:"High"}
    result = mapping[pred]

    st.subheader("Hasil Prediksi")

    # Tampilkan hasil dengan warna
    if result == "Low":
        st.success(f"Priority: **{result}** ⚪👌")
    elif result == "Medium":
        st.warning(f"Priority: **{result}** 🟡😯")
    else:
        st.error(f"Priority: **{result}** 🔴😭")

    # Probabilitas prediksi
    st.write("Probabilitas Prediksi:")
    st.bar_chart(pd.DataFrame({
        "Priority": ["Low", "Medium", "High"],
        "Probability": probs
    }).set_index("Priority"))

    # === Feature Importance (Top 10) ===
    st.subheader("Top 10 Feature Importance")
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(10)

    fig, ax = plt.subplots()
    ax.barh(feat_imp["Feature"], feat_imp["Importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    st.pyplot(fig)

# === Visualisasi tambahan (statis) ===
st.subheader("Distribusi Kelas Ticket (Dataset)")
class_dist = pd.Series([0.50, 0.35, 0.15], index=["Low","Medium","High"])
fig1, ax1 = plt.subplots()
ax1.pie(class_dist, labels=class_dist.index, autopct="%.1f%%", startangle=90)
st.pyplot(fig1)

# === Footer ===
st.markdown("---")
st.caption("Created by Indra Laksana | Final Project Data Science")
