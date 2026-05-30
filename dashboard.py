# ============================================================
#  Predictive Analytics Dashboard
#  Mahmoud Abdelwahab Shaaban — Resume Project #2
# ============================================================
#
#  SETUP (run once in your terminal):
#  py -m pip install streamlit plotly pandas numpy scikit-learn
#
#  HOW TO RUN:
#  streamlit run dashboard.py
#
#  It will open automatically in your browser!
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Analytics Dashboard",
    page_icon="💳",
    layout="wide"
)

# ── Generate Realistic Sample Data ────────────────────────────
@st.cache_data
def generate_data(n_months=24, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.today(), periods=n_months, freq="MS")

    # Revenue with upward trend + seasonality
    trend     = np.linspace(80000, 160000, n_months)
    seasonal  = 15000 * np.sin(np.linspace(0, 4*np.pi, n_months))
    noise     = np.random.normal(0, 5000, n_months)
    revenue   = trend + seasonal + noise

    # Transactions
    transactions = (revenue / np.random.uniform(45, 55, n_months)).astype(int)

    # Customers
    customers = np.cumsum(np.random.randint(200, 600, n_months)) + 3000

    # Churn rate (declining over time — good sign)
    churn = np.linspace(8.5, 3.2, n_months) + np.random.normal(0, 0.4, n_months)

    # Fraud rate
    fraud_rate = np.random.uniform(0.08, 0.25, n_months)

    df = pd.DataFrame({
        "Month":        dates,
        "Revenue":      revenue.round(2),
        "Transactions": transactions,
        "Customers":    customers,
        "Churn_Rate":   churn.round(2),
        "Fraud_Rate":   fraud_rate.round(3),
        "Avg_Transaction": (revenue / transactions).round(2)
    })

    # Category breakdown
    categories = ["E-commerce", "Retail POS", "Mobile Pay", "Subscriptions", "International"]
    cat_data = []
    for _, row in df.iterrows():
        splits = np.random.dirichlet(np.ones(5)) * row["Revenue"]
        for cat, val in zip(categories, splits):
            cat_data.append({"Month": row["Month"], "Category": cat, "Revenue": round(val, 2)})

    return df, pd.DataFrame(cat_data)

df, cat_df = generate_data()

# ── Sidebar Controls ──────────────────────────────────────────
st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.markdown("---")

months_back = st.sidebar.slider("Months to display", 3, 24, 12)
show_forecast = st.sidebar.toggle("Show Revenue Forecast", value=True)
selected_cats = st.sidebar.multiselect(
    "Payment Categories",
    options=cat_df["Category"].unique().tolist(),
    default=cat_df["Category"].unique().tolist()
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Built by:** Mahmoud Abdelwahab Shaaban")
st.sidebar.markdown("**Project:** Analytics Dashboard")

# Filter data
df_filtered  = df.tail(months_back).copy()
cat_filtered = cat_df[
    (cat_df["Month"].isin(df_filtered["Month"])) &
    (cat_df["Category"].isin(selected_cats))
]

# ── Header ────────────────────────────────────────────────────
st.title("💳 Financial Analytics Dashboard")
st.markdown("Real-time payment analytics and revenue intelligence platform")
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total_rev   = df_filtered["Revenue"].sum()
total_trans = df_filtered["Transactions"].sum()
latest_cust = df_filtered["Customers"].iloc[-1]
avg_churn   = df_filtered["Churn_Rate"].mean()
avg_fraud   = df_filtered["Fraud_Rate"].mean()

prev = df.iloc[-(months_back+1):-1] if months_back < len(df) else df_filtered
prev_rev = prev["Revenue"].sum()
rev_delta = ((total_rev - prev_rev) / prev_rev * 100) if prev_rev else 0

col1.metric("Total Revenue",      f"${total_rev:,.0f}",    f"{rev_delta:+.1f}%")
col2.metric("Total Transactions", f"{total_trans:,}",      f"+{np.random.randint(3,12)}%")
col3.metric("Active Customers",   f"{latest_cust:,}",      f"+{np.random.randint(200,500)}")
col4.metric("Avg Churn Rate",     f"{avg_churn:.1f}%",     f"{-abs(np.random.uniform(0.2,0.8)):.1f}%")
col5.metric("Fraud Rate",         f"{avg_fraud:.3f}%",     f"{-abs(np.random.uniform(0.01,0.05)):.3f}%")

st.markdown("---")

# ── Revenue Trend + Forecast ──────────────────────────────────
st.subheader("📈 Revenue Trend & Forecast")

fig_rev = go.Figure()

# Actual revenue
fig_rev.add_trace(go.Scatter(
    x=df_filtered["Month"], y=df_filtered["Revenue"],
    mode="lines+markers", name="Actual Revenue",
    line=dict(color="#2563EB", width=2.5),
    marker=dict(size=6)
))

if show_forecast:
    # Simple linear regression forecast (3 months ahead)
    X = np.arange(len(df_filtered)).reshape(-1, 1)
    y = df_filtered["Revenue"].values
    model = LinearRegression().fit(X, y)

    future_X   = np.arange(len(df_filtered), len(df_filtered)+3).reshape(-1,1)
    future_y   = model.predict(future_X)
    future_dates = pd.date_range(df_filtered["Month"].iloc[-1], periods=4, freq="MS")[1:]

    fig_rev.add_trace(go.Scatter(
        x=list(df_filtered["Month"].iloc[-1:]) + list(future_dates),
        y=list(df_filtered["Revenue"].iloc[-1:]) + list(future_y),
        mode="lines+markers", name="Forecast",
        line=dict(color="#F59E0B", width=2, dash="dash"),
        marker=dict(size=6, symbol="diamond")
    ))

    # Confidence band
    std = np.std(y - model.predict(X))
    fig_rev.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(future_y + 1.5*std) + list((future_y - 1.5*std)[::-1]),
        fill="toself", fillcolor="rgba(245,158,11,0.1)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence Band"
    ))

fig_rev.update_layout(
    height=350, margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis_tickprefix="$", yaxis_tickformat=",.0f",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig_rev, use_container_width=True)

# ── Category Breakdown + Churn ────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("💰 Revenue by Payment Category")
    fig_cat = px.bar(
        cat_filtered, x="Month", y="Revenue", color="Category",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_cat.update_layout(
        height=320, margin=dict(l=0,r=0,t=10,b=0),
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_cat, use_container_width=True)

with col_b:
    st.subheader("📉 Customer Churn Rate Over Time")
    fig_churn = go.Figure()
    fig_churn.add_trace(go.Scatter(
        x=df_filtered["Month"], y=df_filtered["Churn_Rate"],
        mode="lines+markers", fill="tozeroy",
        line=dict(color="#EF4444", width=2),
        fillcolor="rgba(239,68,68,0.1)", name="Churn Rate"
    ))
    fig_churn.update_layout(
        height=320, margin=dict(l=0,r=0,t=10,b=0),
        yaxis_ticksuffix="%",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_churn, use_container_width=True)

# ── Transactions + Fraud ──────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("🔢 Monthly Transaction Volume")
    fig_trans = px.bar(
        df_filtered, x="Month", y="Transactions",
        color_discrete_sequence=["#2563EB"]
    )
    fig_trans.update_layout(
        height=300, margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_trans, use_container_width=True)

with col_d:
    st.subheader("🛡️ Fraud Rate Monitoring")
    fig_fraud = go.Figure()
    fig_fraud.add_trace(go.Scatter(
        x=df_filtered["Month"], y=df_filtered["Fraud_Rate"],
        mode="lines+markers",
        line=dict(color="#8B5CF6", width=2),
        marker=dict(size=6), name="Fraud Rate"
    ))
    fig_fraud.add_hline(y=0.20, line_dash="dash", line_color="red",
                        annotation_text="Alert threshold (0.20%)")
    fig_fraud.update_layout(
        height=300, margin=dict(l=0,r=0,t=10,b=0),
        yaxis_ticksuffix="%",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_fraud, use_container_width=True)

# ── Raw Data Table ────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 View Raw Data"):
    st.dataframe(
        df_filtered.style.format({
            "Revenue": "${:,.2f}",
            "Transactions": "{:,}",
            "Customers": "{:,}",
            "Churn_Rate": "{:.2f}%",
            "Fraud_Rate": "{:.3f}%",
            "Avg_Transaction": "${:.2f}"
        }),
        use_container_width=True
    )
    st.download_button(
        "⬇️ Download CSV",
        df_filtered.to_csv(index=False),
        "analytics_data.csv",
        "text/csv"
    )

st.markdown("---")
st.caption("Dashboard built with Python · Streamlit · Plotly · scikit-learn | Mahmoud Abdelwahab Shaaban")
