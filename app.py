import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load dataset for infographics
df = pd.read_csv("beer-servings.csv")

st.title("Alcohol Consumption Analysis")

st.subheader("Dataset Overview")
st.write(df.head())

st.subheader("Average Alcohol Servings by Continent")
continent_avg = df.groupby("continent")[["beer_servings","spirit_servings","wine_servings"]].mean()
st.bar_chart(continent_avg)

st.subheader("Total Alcohol Consumption Distribution")
st.bar_chart(df["total_litres_of_pure_alcohol"])

# Load the trained model
model = pickle.load(open("alcohol_model.pkl","rb"))

st.title("Alcohol Consumption Prediction")

# Numerical inputs
beer = st.number_input("Beer Servings")
spirit = st.number_input("Spirit Servings")
wine = st.number_input("Wine Servings")

# Dropdown for continent
continent = st.selectbox(
    "Select Continent",
    ["Asia","Europe","North America","Oceania","South America"]
)

# Map continent to one-hot encoding
continent_data = {
    "Asia":[1,0,0,0,0],
    "Europe":[0,1,0,0,0],
    "North America":[0,0,1,0,0],
    "Oceania":[0,0,0,1,0],
    "South America":[0,0,0,0,1]
}

continent_values = continent_data[continent]

# Combine inputs
# Add extra 0 to match the number of features your model expects
input_data = np.array([[beer, spirit, wine] + continent_values + [0]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Alcohol Consumption: {prediction[0]:.2f}")
