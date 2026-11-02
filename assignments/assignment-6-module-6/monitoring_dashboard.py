import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px

st.set_page_config(page_title="Fraud Model Monitor", layout="wide")
st.title("📊 Production Monitoring Dashboard")
st.warning("⚠️ WARNING: This dashboard contains SIMULATED data for demonstration purposes.")

conn = sqlite3.connect("logs/predictions.db")
df = pd.read_sql("SELECT * FROM predictions", conn)
conn.close()

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Requests", len(df))
col2.metric("Avg Latency", f"{df['latency_ms'].mean():.1f} ms")
col3.metric("Fraud Rate (Flagged)", f"{(df['decision'] == 'Fraud').mean() * 100:.1f}%")
col4.metric("Avg Fraud Prob", f"{df['fraud_prob'].mean():.2f}")

# Charts
st.subheader("Prediction Volume & Fraud Probability Over Time")
daily = df.groupby('date').agg(
    requests=('id', 'count'),
    avg_prob=('fraud_prob', 'mean')
).reset_index()

fig1 = px.bar(daily, x='date', y='requests', title="Requests per Day")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.line(daily, x='date', y='avg_prob', title="Average Predicted Fraud Probability")
st.plotly_chart(fig2, use_container_width=True)