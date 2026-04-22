import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="Fraud Monitor", page_icon="🛡️", layout="wide")

# Sidebar
st.sidebar.title("Navigation & Filters")
page = st.sidebar.radio("Go to", ["Live Monitor", "Analytics & Trends", "Transaction Analyzer", "Model Performance"])

st.sidebar.markdown("---")
model_selector = st.sidebar.selectbox("Active Model", ["XGBoost (Tuned)", "LSTM", "Random Forest"])
risk_threshold = st.sidebar.slider("Risk Threshold", 0.1, 0.9, 0.42)

# Load Data Helper
@st.cache_data(ttl=10)
def load_logs():
    log_path = Path("../logs/predictions.json")
    if not log_path.exists(): return pd.DataFrame()
    with open(log_path, "r") as f:
        return pd.DataFrame(json.load(f))

df_logs = load_logs()

if not df_logs.empty:
    st.sidebar.download_button("Download Logs (CSV)", df_logs.to_csv(index=False), "fraud_logs.csv", "text/csv")

if page == "Live Monitor":
    st.title("🔴 Live Transaction Monitor")
    st.markdown("*Auto-refreshing every 10 seconds...*")
    
    if not df_logs.empty:
        latest_100 = df_logs.tail(100).iloc[::-1]
        
        col1, col2, col3, col4 = st.columns(4)
        total_tx = len(df_logs)
        fraud_tx = len(df_logs[df_logs['prediction'] == 1])
        
        col1.metric("Total Transactions Today", f"{total_tx:,}")
        col2.metric("Fraud Detected", f"{fraud_tx:,} ({(fraud_tx/total_tx)*100:.2f}%)")
        col3.metric("Avg Fraud Probability", f"{df_logs['fraud_probability'].mean():.4f}")
        col4.metric("Highest Risk Amount", f"${latest_100[latest_100['risk_level']=='HIGH']['Amount'].max():.2f}" if 'Amount' in latest_100 else "N/A")
        
        def color_risk(val):
            color = 'red' if val == 'HIGH' else 'orange' if val == 'MEDIUM' else 'green'
            return f'color: {color}'
            
        st.dataframe(latest_100.style.map(color_risk, subset=['risk_level']), use_container_width=True)
    else:
        st.info("No transactions logged yet.")
        
    time.sleep(10)
    st.rerun()

elif page == "Analytics & Trends":
    st.title("📊 Analytics & Trends")
    if not df_logs.empty:
        df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
        df_logs['hour'] = df_logs['timestamp'].dt.floor('H')
        
        col1, col2 = st.columns(2)
        
        hourly_fraud = df_logs.groupby('hour')['prediction'].mean().reset_index()
        fig_line = px.line(hourly_fraud, x='hour', y='prediction', title='Fraud Rate Over Time (Hourly)')
        col1.plotly_chart(fig_line, use_container_width=True)
        
        fig_bar = px.bar(df_logs['risk_level'].value_counts().reset_index(), x='risk_level', y='count', title='Transactions by Risk Level', color='risk_level', color_discrete_map={'LOW':'green', 'MEDIUM':'orange', 'HIGH':'red'})
        col2.plotly_chart(fig_bar, use_container_width=True)
        
        fig_donut = px.pie(df_logs, names='prediction', title='Fraud vs Legitimate Ratio', hole=0.4)
        col1.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("No data available for analytics.")

elif page == "Transaction Analyzer":
    st.title("🔍 Single Transaction Analyzer")
    with st.form("tx_form"):
        tx_id = st.text_input("Transaction ID", "TXN-9999")
        amount = st.number_input("Amount ($)", min_value=0.0, value=150.0)
        submitted = st.form_submit_button("Analyze Transaction")
        
        if submitted:
            # Mock response for demo
            res_data = {"prediction": 1, "fraud_probability": 0.88, "risk_level": "HIGH"}
            
            if res_data['prediction'] == 1:
                st.error(f"🚨 FRAUD DETECTED (Risk: {res_data['risk_level']})")
            else:
                st.success("✅ LEGITIMATE TRANSACTION")
                
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = res_data['fraud_probability'],
                title = {'text': "Fraud Probability"},
                gauge = {'axis': {'range': [0, 1]}, 'bar': {'color': "red" if res_data['prediction']==1 else "green"}}
            ))
            st.plotly_chart(fig)

elif page == "Model Performance":
    st.title("📈 Model Performance Tracker")
    metrics_path = Path("../models/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        st.json(metrics)
    else:
        st.info("Metrics file not found. Run the training pipeline to generate it.")
