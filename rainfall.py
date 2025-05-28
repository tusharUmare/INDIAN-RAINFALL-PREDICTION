
import streamlit as st

st.set_page_config(page_title="India Rainfall Dashboard", layout="wide")

st.title("🌧️ Welcome to Indian Rainfall Analysis Dashboard")
st.markdown("Use the left sidebar to navigate through **EDA**, **ML Models**, **Forecasting**, and **Clustering**.")
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Monsoon_India_2016.svg/800px-Monsoon_India_2016.svg.png", width=700)
import streamlit as st

theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown(
        """
        <style>
        body {
            background-color: #111;
            color: #eee;
        }
        </style>
        """, unsafe_allow_html=True
    )
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("DataSets/rainfaLLIndia.csv")
df['JUN-SEP'] = df[['JUN', 'JUL', 'AUG', 'SEP']].mean(axis=1)

st.title("📊 Exploratory Data Analysis")

st.subheader("Average Monsoon Rainfall Over Years")
fig1, ax1 = plt.subplots()
sns.lineplot(data=df.groupby('YEAR')['JUN-SEP'].mean().reset_index(), x='YEAR', y='JUN-SEP', ax=ax1)
st.pyplot(fig1)

st.subheader("Distribution of Monsoon Rainfall")
fig2, ax2 = plt.subplots()
sns.histplot(df['JUN-SEP'], bins=30, kde=True, ax=ax2)
st.pyplot(fig2)

st.subheader("Monthly Rainfall Distribution")
fig3, ax3 = plt.subplots()
sns.boxplot(data=df[['JUN', 'JUL', 'AUG', 'SEP']], ax=ax3)
st.pyplot(fig3)

st.subheader("Correlation Matrix")
fig4, ax4 = plt.subplots()
sns.heatmap(df[['JUN', 'JUL', 'AUG', 'SEP']].corr(), annot=True, cmap='coolwarm', ax=ax4)
st.pyplot(fig4)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("DataSets/rainfaLLIndia.csv")
df['JUN-SEP'] = df[['JUN', 'JUL', 'AUG', 'SEP']].mean(axis=1)

df_avg_year = df.groupby('YEAR')['JUN-SEP'].mean().reset_index()

X_rf = df_avg_year[['YEAR']].values
y_rf = df_avg_year['JUN-SEP'].values

model = RandomForestRegressor()
model.fit(X_rf, y_rf)

next_year = df_avg_year['YEAR'].max() + 1
next_year_pred = model.predict([[next_year]])[0]

df_avg_year['Prediction'] = model.predict(X_rf)

st.title("📈 Forecasting Monsoon Rainfall")
fig, ax = plt.subplots()
ax.plot(df_avg_year['YEAR'], df_avg_year['JUN-SEP'], label="Actual", marker='o')
ax.plot(df_avg_year['YEAR'], df_avg_year['Prediction'], label="Prediction", marker='x')
ax.scatter([next_year], [next_year_pred], color='red', label=f"{next_year} Forecast")
ax.axvline(next_year, linestyle='--', color='gray')
st.pyplot(fig)

st.success(f"📌 Predicted Rainfall for {next_year}: {next_year_pred:.2f} mm")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv("DataSets/rainfaLLIndia.csv")
df['JUN-SEP'] = df[['JUN', 'JUL', 'AUG', 'SEP']].mean(axis=1)

subdiv_avg = df.groupby('subdivision')['JUN-SEP'].mean().reset_index()
kmeans = KMeans(n_clusters=4, random_state=42)
subdiv_avg['Cluster'] = kmeans.fit_predict(subdiv_avg[['JUN-SEP']])

st.title("🔍 Clustering Subdivisions by Rainfall")
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=subdiv_avg.sort_values(by='JUN-SEP'), x='subdivision', y='JUN-SEP', hue='Cluster', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)
