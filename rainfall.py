import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.cluster import KMeans

# Page Config
st.set_page_config(page_title="India Rainfall Dashboard", layout="wide")

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("../DataSets/rainfaLLIndia.csv")
    df['JUN-SEP'] = df[['JUN', 'JUL', 'AUG', 'SEP']].mean(axis=1)
    df['YoY_CHANGE'] = df.groupby('subdivision')['JUN-SEP'].diff()
    df['LAG1'] = df.groupby('subdivision')['JUN-SEP'].shift(1)
    df['YoY_CHANGE'] = df['YoY_CHANGE'].fillna(211)
    df['LAG1'] = df['LAG1'].fillna(264)
    return df

df = load_data()

# Sidebar
st.sidebar.title("Options")
selected_subdivision = st.sidebar.selectbox("Select Subdivision", df['subdivision'].unique())

# Encode subdivision
label = LabelEncoder()
df['subdivision_code'] = label.fit_transform(df['subdivision'])

# Subset data
df_sub = df[df['subdivision'] == selected_subdivision]

# Section: Overview
st.title("🌧️ Indian Rainfall Analysis & Forecasting")
st.markdown(f"### 📍 Subdivision: **{selected_subdivision}**")

# EDA Plots
st.header("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=df.groupby('YEAR')['JUN-SEP'].mean().reset_index(), x='YEAR', y='JUN-SEP', ax=ax1)
    ax1.set_title("Average Monsoon Rainfall Over Years")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.histplot(df['JUN-SEP'], bins=30, kde=True, ax=ax2)
    ax2.set_title("Distribution of Monsoon Rainfall")
    st.pyplot(fig2)

fig3, ax3 = plt.subplots()
sns.boxplot(data=df[['JUN', 'JUL', 'AUG', 'SEP']], ax=ax3)
ax3.set_title("Monthly Rainfall Distribution")
st.pyplot(fig3)

# Correlation
st.subheader("🌐 Correlation Matrix")
fig_corr, ax_corr = plt.subplots()
sns.heatmap(df[['JUN', 'JUL', 'AUG', 'SEP']].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
st.pyplot(fig_corr)

# Section: Machine Learning
st.header("🤖 Machine Learning — Rainfall Prediction")

# Prepare data
X = df.drop(['LAG1'], axis=1)._get_numeric_data().values
y = df['LAG1'].values

tscv = TimeSeriesSplit(n_splits=5, test_size=30)
scaler = StandardScaler()

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    xtrain, xtest = X[train_idx], X[test_idx]
    ytrain, ytest = y[train_idx], y[test_idx]
    x_train = scaler.fit_transform(xtrain)
    x_test = scaler.transform(xtest)

    model = LinearRegression()
    model.fit(x_train, ytrain)
    ypred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(ytest, ypred))
    r2 = r2_score(ytest, ypred)

    fig_pred, ax_pred = plt.subplots()
    ax_pred.plot(ytest, label='Actual', marker='o')
    ax_pred.plot(ypred, label='Predicted', marker='x')
    ax_pred.set_title(f'Fold {fold+1} — RMSE: {rmse:.2f}, R²: {r2:.2f}')
    ax_pred.legend()
    st.pyplot(fig_pred)
    break  # Display only one fold for simplicity

# Random Forest
st.subheader("🌲 Random Forest Accuracy")
model_rf = RandomForestRegressor()
model_rf.fit(x_train, ytrain)
rf_test_score = r2_score(ytest, model_rf.predict(x_test))
rf_train_score = r2_score(ytrain, model_rf.predict(x_train))
st.write(f"**Training Accuracy:** {rf_train_score*100:.2f}%")
st.write(f"**Testing Accuracy:** {rf_test_score*100:.2f}%")

# XGBoost
st.subheader("⚡ XGBoost Accuracy")
model_xg = XGBRegressor()
model_xg.fit(x_train, ytrain)
xg_test_score = r2_score(ytest, model_xg.predict(x_test))
st.write(f"**Testing Accuracy:** {xg_test_score*100:.2f}%")

# Clustering
st.header("🔍 Clustering Based on Rainfall")

subdiv_avg = df.groupby('subdivision')['JUN-SEP'].mean().reset_index()
kmeans = KMeans(n_clusters=4, random_state=42)
subdiv_avg['Cluster'] = kmeans.fit_predict(subdiv_avg[['JUN-SEP']])

fig_clust, ax_clust = plt.subplots(figsize=(12, 6))
sns.barplot(x='subdivision', y='JUN-SEP', hue='Cluster', data=subdiv_avg.sort_values('JUN-SEP'), ax=ax_clust)
ax_clust.set_xticklabels(ax_clust.get_xticklabels(), rotation=90)
ax_clust.set_title("Subdivision Clusters by Avg Monsoon Rainfall")
st.pyplot(fig_clust)

# Forecasting + Next Year
st.header("📈 Forecasting Monsoon Rainfall")

df_avg_year = df.groupby('YEAR')['JUN-SEP'].mean().reset_index()
X_rf = df_avg_year[['YEAR']].values
y_rf = df_avg_year['JUN-SEP'].values
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_rf, y_rf)
df_avg_year['Predicted'] = rf.predict(X_rf)

# Predict next year
next_year = df_avg_year['YEAR'].max() + 1
next_year_pred = rf.predict([[next_year]])[0]

fig_forecast, ax_forecast = plt.subplots()
ax_forecast.plot(df_avg_year['YEAR'], df_avg_year['JUN-SEP'], label='Actual', marker='o')
ax_forecast.plot(df_avg_year['YEAR'], df_avg_year['Predicted'], label='Prediction (RF)', marker='x')
ax_forecast.axvline(x=next_year, linestyle='--', color='gray')
ax_forecast.scatter([next_year], [next_year_pred], color='red', label=f"Forecast {next_year}")
ax_forecast.set_title("🇮🇳 Avg Monsoon Rainfall Trend with RF Prediction")
ax_forecast.set_xlabel("Year")
ax_forecast.set_ylabel("Avg Rainfall (JUN-SEP)")
ax_forecast.legend()
st.pyplot(fig_forecast)

st.success(f"📌 Forecasted Rainfall for {next_year}: **{next_year_pred:.2f} mm**")

# CSV Download
st.header("📥 Download Processed Dataset")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV",
    data=csv,
    file_name='processed_rainfall_data.csv',
    mime='text/csv'
)

