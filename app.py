# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -------------------------------
# 1. Load Data
# -------------------------------
st.title("Sales Price Prediction & Analytics Dashboard")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    st.warning("Using default dataset from repo")
    data = pd.read_csv("SuperMarket Analysis.csv")  # make sure this is in your GitHub repo

# -------------------------------
# 2. Preprocessing
# -------------------------------
data = data.dropna()
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Keep only numeric + necessary categorical columns for model
numeric_features = ['Quantity', 'Unit price', 'Tax 5%', 'cogs', 'gross income', 'Rating']
categorical_features = ['Product line', 'Gender', 'Branch', 'City', 'Customer type']

data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# -------------------------------
# 3. Train Model
# -------------------------------
X = data[numeric_features]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 4. Data Overview
# -------------------------------
st.subheader("Sales Data Overview")
st.dataframe(data.head())

# -------------------------------
# 5. Graphs
# -------------------------------
st.subheader("📊 Sales Data Visualizations")

# Sales over Time
if 'Date' in data.columns:
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data['Sales'])
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title("Sales Trend Over Time")
    st.pyplot(fig)

# Sales by City
if any(col.startswith('City_') for col in data.columns):
    city_cols = [col for col in data.columns if col.startswith('City_')]
    city_sales = data[city_cols].multiply(data['Sales'], axis=0).sum().sort_values(ascending=False)
    fig2, ax2 = plt.subplots()
    sns.barplot(x=city_sales.index.str.replace('City_', ''), y=city_sales.values, ax=ax2)
    ax2.set_ylabel("Total Sales")
    ax2.set_title("Sales by City")
    st.pyplot(fig2)

# -------------------------------
# 6. Predict Sales
# -------------------------------
st.subheader("Predict Sales")
st.write("Enter values for numeric features:")

input_dict = {}
for feature in numeric_features:
    if feature == 'Quantity':
        input_dict[feature] = st.number_input(feature, min_value=1, value=1)
    else:
        input_dict[feature] = st.number_input(feature, min_value=0.0, value=float(data[feature].mean()))

input_df = pd.DataFrame([input_dict])

predicted_sale = model.predict(input_df)
st.success(f"Predicted Sales: {predicted_sale[0]:.2f}")
