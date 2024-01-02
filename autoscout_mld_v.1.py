# import libs
import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import pickle

# Title with a different color scheme
st.markdown("""
<div style="background-color: #3498db; padding: 10px">
    <h2 style="color: white; text-align: center;">Car Price Prediction App</h2>
</div>
""", unsafe_allow_html=True)
st.markdown("")

# Buttons layout with adjusted colors
buttons_row = st.columns([1, 1, 1])  # Adjusting column widths

# Initialize session_state if it's the first run
if 'data_button_clicked' not in st.session_state:
    st.session_state.data_button_clicked = False

# Data button 
data_button_key = "data_button"  # Unique key
if buttons_row[0].button("Data", key=data_button_key, help="Click to view or hide data"):
    st.session_state.data_button_clicked = not st.session_state.data_button_clicked

# Initialize 'info_button_clicked' and 'links_button_clicked' if it's the first run
if 'info_button_clicked' not in st.session_state:
    st.session_state.info_button_clicked = False
if 'links_button_clicked' not in st.session_state:
    st.session_state.links_button_clicked = False

# Information and Links buttons with different colors
info_button, links_button = st.columns([1, 0.111])  # Adjusting column widths

info_button_key = "info_button"  # Unique key
if info_button.button("Information", key=info_button_key, help="Click for additional information"):
    st.session_state.info_button_clicked = not st.session_state.info_button_clicked

links_button_key = "links_button"  # Unique key
if links_button.button("Links", key=links_button_key, help="Click for useful links"):
    st.session_state.links_button_clicked = not st.session_state.links_button_clicked

# Sidebar title with a different color
st.sidebar.markdown("""
<div style="background-color: #27ae60; padding: 10px">
    <h2 style="color: white; text-align: center;">Car Price Prediction</h2>
</div>
""", unsafe_allow_html=True)

# Load data
selected_algorithm = st.sidebar.selectbox("Random Forest", [""], index=0)
df = pd.read_csv("deploy_data.csv")

# Display training data
st.header("Training Dataframe is below")
st.markdown("---")
st.write(df.sample(5))

# Sidebar inputs with adjusted styling
make_model = st.sidebar.selectbox("Select Auto Brand - Model", df["make_model"].unique(), index=1)
gearbox = st.sidebar.selectbox("Select Gearbox", df["gearbox"].unique(), index=1)
drivetrain = st.sidebar.selectbox("Select Drivetrain", df["drivetrain"].unique(), index=1)
power_kw = st.sidebar.number_input("Enter Power (in kW)", min_value=df["power_kW"].min(), max_value=df["power_kW"].max(), value=df["power_kW"].mode().iloc[0], step=1.0)
age=st.sidebar.selectbox("Age of your car:",(0,1,2,3))
# Load machine learning model
model_rf = pickle.load(open("best_model", "rb"))

# Prepare input for prediction
my_dict = {
    "power_kW": power_kw,
    "age": age,
    # Add other features...
}
df_input = pd.DataFrame.from_dict([my_dict])
df_input = df_input[["make_model", "gearbox", "drivetrain", "power_kW", "age", "empty_weight", "mileage", "gears", "cons_avg", "co_emissions"]]

# Display user inputs
st.header("The values you selected are below")
st.markdown("---")
st.table(df_input)

# Car Prediction title with different styling
st.title("Car Prediction")

# Predict button with a different color
if st.button("Predict", help="Click to predict car price"):
    result = None
    if selected_algorithm == "Random Forest":
        result = model_rf.predict(df_input)[0]

    if result is not None:
        st.success(f"With {selected_algorithm}, Car Price is **{round(result, 0)}**")
    else:
        st.write("Press **Predict** button to display the result!")
