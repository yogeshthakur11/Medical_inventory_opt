import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st

# Load the dataset
df = pd.read_csv(r"C:\Users\Yogesh Thakur\opt.csv")

# Convert the "Dateofbill" column to a datetime object
df["Dateofbill"] = pd.to_datetime(df["Dateofbill"])

# Group the dataframe by DrugName and week, and sum the quantity sold for each week
df_grouped = df.groupby(["DrugName", pd.Grouper(key="Dateofbill", freq="W-MON")])["Quantity"].sum().reset_index()

# Fill missing values for DrugName with 'Unknown'
df[['DrugName']] = df[['DrugName']].fillna(value='Unknown')

# Get the top 100 best-selling drugs
top_drugs = df_grouped.groupby("DrugName")["Quantity"].sum().sort_values(ascending=False)[:752].index.tolist()

# Create a dataframe with only the top 100 best-selling drugs
df_top = df_grouped[df_grouped["DrugName"].isin(top_drugs)]

# Pivot the dataframe to have the drug names as columns and the dates as rows
df_pivot = df_top.pivot(index="Dateofbill", columns="DrugName", values="Quantity")

df_pivot.fillna(0, inplace=True)

# Calculate the 4-week Moving Average for each drug
df_ma = df_pivot.rolling(window=4).mean()

df_ma.fillna(0, inplace=True)

# Forecast the quantity for the next 53 weeks using the MA model
forecast = pd.DataFrame(columns=top_drugs)
for drug in top_drugs:
    forecast[drug] = pd.concat([df_ma[drug], df_ma[drug][-3:].rolling(window=3).mean()]).tail(53).reset_index(drop=True)
    
forecast.fillna(0, inplace=True)

# Define a function to plot the forecasted values week-wise
def plot_forecast(drug, df_pivot, forecast):
    weeks = pd.date_range(start=df_top["Dateofbill"].max(), periods=53, freq="W-MON")
    plt.figure(figsize=(10, 5))
    plt.plot(df_pivot[drug])
    plt.plot(weeks, forecast[drug])
    plt.title(drug)
    plt.xlabel("Date")
    plt.ylabel("Quantity")
    plt.legend(["Actual", "Forecast"])
    return plt

# Create a Streamlit app
st.title("Drug Quantity Forecasting")
st.sidebar.title("Settings")

# Add a slider for the number of weeks to forecast
num_weeks = st.sidebar.slider("Number of weeks to forecast", min_value=1, max_value=104, value=53)

# Plot the forecasted values for each drug
for drug in top_drugs:
    plot = plot_forecast(drug, df_pivot, forecast)
    st.pyplot(plot)

# Print the quantity by week-wise for the next 53 weeks
forecast_float = forecast.astype(float)
st.write("Forecasted quantities (week-wise):")
st.write(forecast_float.tail(num_weeks))




