import streamlit as st
import pandas as pd
from husggingface_hub import hf_hub_download
import joblib
model_path = hf_hub_download(repo_id = "Lokeshnathy/best-model-boston-house-price-prediction",
                             filename = "best_house_price_predictor_model_v1.joblib")
model = joblib.load(model_path)
# Streamlit UI
st.title("Boston Housing🏡 Property Price💵 Predictor")
st.subheader("Online Property Price Prediction App.")
st.write("This App is an internal tool kit meant for the Boston Housing corp. management to check the price of a housing property.")
st.write("Kindly enter the property details to start predicting the prices.")
CRIM = st.number_input("Crime Rate of the town(per capita)", min_value=0.00001,max_value=50.00000,step=0.00001,value=0.03768)
ZN = st.number_input("Proportion of Residential Land zoned for lots over 25K sft", min_value=0.0,max_value=200.0,step=0.1,value=80.0)
INDUS =st.number_input("Non-Retail Business land proportion.",min_value=0.00,max_value=50.00,step=0.01,value=1.52)
CHAS = st.selectbox("Whether falls in Charles River tract.", ['yes','no'])
NX = st.number_input("Nitric Oxide Level(ppm)",min_value=0.000,max_value=1.000,step=0.001,value=0.404)
RM = st.number_input("Room count per dwelling",min_value=0.000,max_value=10.000,step=0.001,value=7.274)
AGE = st.number_input("Owner-occupied units(prior 1940).",min_value=18.0,max_value=100.0,step=0.1,value=38.3)
DIS = st.number_input("Distance from popular employment centers.",min_value=0.0000,max_value=25.0000,step=0.0001,value=7.3090)
RAD = st.number_input("Access to Radial Highways.",min_value=0.0,max_value=50.0,step=1.0,value=2.0)
TAX = st.number_input("Property Tax per $10K",min_value=1.0,max_value=1000.0,step=1.0,value=329.0)
PTRATIO = st.number_input("Pupil to Teacher Ratio",min_value=0.0,max_value=25.0,value=0.1)
LSTAT = st.number_input("Percent of Lower Status Population",min_value=0.00,max_value=50.00,value=3.81)
# Converting user inputs into a DataFrame
input_data = pd.DataFrame([{
    'Crime Rate of the town(per capita)': CRIM,
    'Proportion of Residential Land zoned for lots over 25K sft': ZN,
    'Non-Retail Business land proportion.': INDUS,
    'Whether falls in Charles River tract.': 1 if CHAS=="yes" else 0,
    'Nitric Oxide Level(ppm)': NX,
    'Room count per dwelling': RM,
    'Owner-occupied units(prior 1940).': AGE,
    'Distance from popular employment centers.': DIS,
    'Access to Radial Highways.': RAD,
    'Property Tax per $10K"': TAX,
    'Pupil to Teacher Ratio': PTRATIO,
    'Percent of Lower Status Population': LSTAT}])
# Predict button
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Estimated Property Price: **${prediction:,.2f} USD**")
