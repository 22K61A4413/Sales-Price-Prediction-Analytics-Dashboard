#a) Import libraries

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#b) Load data

# Load CSV (use relative path in GitHub repo)
data = pd.read_csv('SuperMarket Analysis.csv')
# Preprocess
data = data.dropna()
data['Date'] = pd.to_datetime(data['Date'])

#c) Encode categorical variables

data = pd.get_dummies(data, columns=['Product line', 'Gender', 'Branch', 'City', 'Customer type'], drop_first=True)

#d) Train model

X = data[['Quantity', 'Unit price']]  # or all numerical features you want
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

#e) Streamlit dashboard

st.title("Sales Analytics Dashboard")

# 1. Show data
st.subheader("Data Overview")
st.dataframe(data.head())

# 2. Graphs
st.subheader("Sales Visualization")
fig, ax = plt.subplots()
ax.plot(data['Date'], data['Sales'])
ax.set_title("Sales Over Time")
st.pyplot(fig)

# 3. Prediction input
st.subheader("Predict Sales")
quantity = st.number_input('Quantity', min_value=1, value=1)
unit_price = st.number_input('Unit Price', min_value=1.0, value=10.0)
predicted_sale = model.predict([[quantity, unit_price]])
st.write("Predicted Sales:", predicted_sale[0])

#f) Optional graphs

# Sales by City
city_sales = data.groupby('City')['Sales'].sum()
fig2, ax2 = plt.subplots()
sns.barplot(x=city_sales.index, y=city_sales.values, ax=ax2)
st.pyplot(fig2)
