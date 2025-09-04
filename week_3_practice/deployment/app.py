import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="praneeth232/playstore_revenue_model", filename="best_playstore_revenue_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Play Store App Revenue Prediction")
st.write("""
This application predicts the expected **ad revenue** of a Play Store application 
based on its characteristics such as category, installs, active users, and screen time.
Please enter the app details below to get a revenue prediction.
""")

# User input
app_category = st.selectbox("App Category", ["FAMILY", "TOOLS", "NEWS AND MAGAZINES", "GAME", "EDUCATION", "OTHERS"])
free_or_paid = st.selectbox("Free or Paid", ["Free", "Paid"])
content_rating = st.selectbox("Content Rating", ["Everyone", "Teen", "Mature 17+", "Adults only 18+"])
screentime_category = st.selectbox("Screen Time Category", ["Low", "Medium", "High"])

app_size = st.number_input("App Size (MB)", min_value=1.0, max_value=1000.0, value=50.0, step=0.1)
price = st.number_input("Price (USD)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
installs = st.number_input("Number of Installs", min_value=0, max_value=100000000, value=1000, step=100)
screen_time = st.number_input("Average Screen Time (minutes)", min_value=0, max_value=500, value=30)
active_users = st.number_input("Active Users", min_value=0, max_value=10000000, value=1000, step=100)
short_ads = st.number_input("Short Ads per Hour", min_value=0, max_value=10, value=2)
long_ads = st.number_input("Long Ads per Hour", min_value=0, max_value=10, value=1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'app_category': app_category,
    'free_or_paid': free_or_paid,
    'content_rating': content_rating,
    'screentime_category': screentime_category,
    'app_size_in_mb': app_size,
    'price_in_usd': price,
    'number_of_installs': installs,
    'average_screen_time': screen_time,
    'active_users': active_users,
    'no_of_short_ads_per_hour': short_ads,
    'no_of_long_ads_per_hour': long_ads
}])

# Predict button
if st.button("Predict Revenue"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Estimated Ad Revenue: **${prediction:,.2f} USD**")
