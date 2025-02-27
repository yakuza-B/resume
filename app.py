from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os
import streamlit as st

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["crop_database"]
collection = db["crop_yield"]

# Fetch data from MongoDB
data = list(collection.find({}, {"_id": 0, "temperature": 1, "pesticides": 1, "crop_yield": 1}))
df = pd.DataFrame(data)

# Check if data exists
if df.empty:
    raise ValueError("No data found in MongoDB! Make sure data is inserted correctly.")

# Define features and target variable
X = df[["temperature", "pesticides"]]
y = df["crop_yield"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "crop_yield_model.pkl")
print("Model trained and saved successfully!")

# Streamlit App
st.title("Crop Yield Prediction")
st.write("Enter temperature and pesticide levels to predict crop yield.")

# User Inputs
temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
pest = st.number_input("Pesticides Used (kg/ha)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

# Load trained model
model = joblib.load("crop_yield_model.pkl")

# Prediction Button
if st.button("Predict Crop Yield"):
    prediction = model.predict([[temp, pest]])[0]
    st.success(f"Predicted Crop Yield: {prediction:.2f} tons/ha")

# GitHub Integration (Commit and Push Script)
os.system("git add app.py")
os.system("git commit -m 'Added Streamlit UI for crop yield prediction' ")
os.system("git push origin main")

print("Script pushed to GitHub!")
