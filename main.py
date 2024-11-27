# imports
import sqlite3
import numpy as np
import pandas as  pd 
from sklearn.linear_model import LinearRegression
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# conecting database
conn = sqlite3.connect('data.sqlite')
warehouse = pd.read_sql_query("SELECT * FROM warehouse", conn)
conn.close()

# loading database
# warehouse

warehouse.replace('', np.nan, inplace=True)
warehouse.dropna(inplace=True)

# data processing 
numeric_cols = warehouse.select_dtypes(include=['float64', 'int64']).columns
warehouse[numeric_cols] = warehouse[numeric_cols].fillna(warehouse[numeric_cols].mean())

# feature selection
X = warehouse[['num_refill_req_l3m', 'transport_issue_l1y', 'Competitor_in_mkt', 'retail_shop_num', 'distributor_num',
          'flood_impacted', 'flood_proof', 'electric_supply', 'dist_from_hub', 'workers_num', 'temp_reg_mach',
          'wh_breakdown_l3m', 'govt_check_l3m']]
y = warehouse['product_wg_ton']

# Ensure y contains only numeric values
y = pd.to_numeric(y, errors='coerce')
y.dropna(inplace=True)

# model building
model = LinearRegression()
model.fit(X, y)

# predict product_wg_ton
prediction = model.predict(X)

# calculate rsquared
SSR = np.sum((prediction - np.mean(y))**2)
SST = np.sum((y - np.mean(y))**2)
rsquared = SSR/SST

# Calculate RMSE, MAE, and MSE
rmse = np.sqrt(mean_squared_error(y, prediction))
mae = mean_absolute_error(y, prediction)
mse = mean_squared_error(y, prediction)

# create streamlit app 
st.title("Optimum Product Weight Predictions")

# create a table
warehouse['Predicted_product_wg_ton'] = prediction
st.subheader("Prediction table")
st.write(warehouse[['Ware_house_ID', 'WH_Manager_ID', 'Predicted_product_wg_ton']])

# Plot the graph for predicted optimum product weight
st.subheader("Predicted Optimum Product Weight for Each Warehouse")
fig, ax = plt.subplots()
ax.scatter(range(len(prediction)), prediction, color='blue', label='Predicted product weight')
ax.set_xlabel('Warehouse Index')
ax.set_ylabel('Optimum Product Weight (tons)')
ax.set_title('Predicted Optimum Product Weight for Each Warehouse')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Display R-squared value, RMSE, MAE, and MSE
st.subheader("Model Evaluation Metrics")
st.write(f"R-squared: {rsquared:.4f}")
st.write(f"RMSE: {rmse:.4f}")
st.write(f"MAE: {mae:.4f}")
st.write(f"MSE: {mse:.4f}")

# Plot the predictions vs actual values
st.subheader('Predictions vs Actual Values')

plt.figure(figsize=(10, 6))
plt.scatter(y, prediction, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
st.pyplot(plt)