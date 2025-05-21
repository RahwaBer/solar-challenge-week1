import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(page_title="Solar Data Dashboard", layout="wide")

@st.cache_data
def load_data():
    df_benin = pd.read_csv(r"..\src\data\Benin_clean.csv")
    df_sierra = pd.read_csv(r"..\src\data\SierraLeone_clean.csv")
    df_togo = pd.read_csv(r"..\src\data\Togo_clean.csv")
    
    df_benin['Country'] = 'Benin'
    df_sierra['Country'] = 'Sierra Leone'
    df_togo['Country'] = 'Togo'

    return pd.concat([df_benin, df_sierra, df_togo], ignore_index=True)

df = load_data()

# Sidebar filters
st.sidebar.title("🔎 Filter Options")
countries = df['Country'].unique().tolist()
selected_countries = st.sidebar.multiselect("Select Countries", countries, default=countries)

metrics = ['GHI', 'DNI', 'DHI']
selected_metric = st.sidebar.selectbox("Select Metric", metrics)

# Filter by country
filtered_df = df[df['Country'].isin(selected_countries)]

# Header
st.title("☀️ Solar Radiation & Environmental Dashboard")
st.markdown("Analyze solar irradiance and environmental measurements across African regions.")

# KPIs
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Avg GHI", round(filtered_df['GHI'].mean(), 2))
with col2:
    st.metric("Avg DNI", round(filtered_df['DNI'].mean(), 2))
with col3:  
    st.metric("Avg DHI", round(filtered_df['DHI'].mean(), 2))

st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

# Boxplot
st.subheader(f"📦 Boxplot of {selected_metric} by Country")
fig1, ax1 = plt.subplots()
sns.boxplot(data=filtered_df, x='Country', y=selected_metric, palette='Set2', ax=ax1)
st.pyplot(fig1)

st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)

# Line Plot
st.subheader("📈 Time Series of Solar Radiation")
filtered_df['Timestamp'] = pd.to_datetime(filtered_df['Timestamp'])
fig2, ax2 = plt.subplots(figsize=(10, 4))
for country in selected_countries:
    country_df = filtered_df[filtered_df['Country'] == country]
    ax2.plot(country_df['Timestamp'], country_df[selected_metric], label=country)
ax2.set_ylabel(f"{selected_metric} (W/m²)")
ax2.legend()
st.pyplot(fig2)

st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)

# Correlation heatmap
st.subheader("🔗 Correlation Heatmap (Selected Countries)")
corr_cols = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB']
fig3, ax3 = plt.subplots()
sns.heatmap(filtered_df[corr_cols].corr(), annot=True, cmap='coolwarm', ax=ax3)
st.pyplot(fig3)

st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)

#Summmary table
summary = {}
for country in selected_countries:
    df_country = filtered_df[filtered_df['Country'] == country]
    stats = df_country[metrics].agg(['mean', 'median', 'std'])
    summary[country] = stats

summary_df = pd.concat(summary, axis=1)
summary_df.columns = summary_df.columns.swaplevel(0, 1)
summary_df = summary_df.sort_index(axis=1, level=0)
summary_df = summary_df.round(2)

# Display summary table in Streamlit
st.subheader("Summary Statistics for GHI, DNI, and DHI by Country")
st.dataframe(summary_df)