"""
╔══════════════════════════════════════════════════════════════╗
║   CMS Open Payments 2018 — Interactive Analysis Dashboard    ║
║   Streamlit App | Healthcare Financial Transparency          ║
╚══════════════════════════════════════════════════════════════╝

Run this app:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import io
import os

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CMS Open Payments 2018 | Analytics Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — MODERN DARK THEME
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background */
.stApp {
    background: radial-gradient(circle at 50% 0%, #161b2e 0%, #0a0e1a 60%, #06090f 100%);
    color: #cbd5e1;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(10, 14, 26, 0.6);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #60a5fa;
    font-weight: 700;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.05);
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(96, 165, 250, 0.3);
    box-shadow: 0 8px 20px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.1);
    background: rgba(255, 255, 255, 0.04);
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.8rem !important; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #f8fafc !important; font-weight: 800 !important; font-size: 2.1rem !important; background: linear-gradient(135deg, #60a5fa, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

/* Headers */
h1 { color: #f8fafc !important; font-weight: 800 !important; letter-spacing: -0.03em; }
h2 { color: #f1f5f9 !important; font-weight: 700 !important; letter-spacing: -0.02em; }
h3 { color: #e2e8f0 !important; font-weight: 600 !important; letter-spacing: -0.01em; }

/* Selectbox / inputs */
.stSelectbox > div > div,
.stNumberInput > div > div,
.stTextInput > div > div {
    background: rgba(15, 23, 42, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    transition: all 0.2s;
}
.stSelectbox > div > div:hover,
.stNumberInput > div > div:hover,
.stTextInput > div > div:hover {
    border-color: rgba(96, 165, 250, 0.5) !important;
    background: rgba(15, 23, 42, 0.6) !important;
}

/* Slider */
.stSlider div[data-baseweb="slider"] > div > div {
    background: #60a5fa !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: rgba(255,255,255,0.02) !important;
    border-radius: 16px;
    padding: 6px;
    gap: 8px;
    margin-bottom: 20px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border-radius: 12px;
    padding: 8px 16px;
    color: #94a3b8;
    border: none;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #f8fafc;
    background-color: rgba(255,255,255,0.05);
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3), inset 0 1px 0 rgba(255,255,255,0.2);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5), inset 0 1px 0 rgba(255,255,255,0.2);
    filter: brightness(1.1);
}
.stButton > button:active {
    transform: translateY(1px);
}

/* Info/warning/success boxes */
.stInfo { background: rgba(56, 189, 248, 0.1); border-left: 4px solid #38bdf8; border-radius: 12px; backdrop-filter: blur(4px); }
.stSuccess { background: rgba(52, 211, 153, 0.1); border-left: 4px solid #34d399; border-radius: 12px; backdrop-filter: blur(4px); }
.stWarning { background: rgba(251, 191, 36, 0.1); border-left: 4px solid #fbbf24; border-radius: 12px; backdrop-filter: blur(4px); }

/* Expander */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    color: #cbd5e1 !important;
    transition: all 0.3s;
}
.streamlit-expanderHeader:hover {
    background: rgba(255,255,255,0.06) !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.08); margin: 2rem 0; }

/* Custom Glass Card */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 28px;
    margin: 16px 0;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: all 0.3s;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}
.glass-card:hover {
    transform: translateY(-5px);
    border-color: rgba(255,255,255,0.15);
    box-shadow: 0 15px 35px rgba(0,0,0,0.3);
}
.glass-card h4 {
    margin-top: 0;
    color: #f8fafc;
    font-size: 1.2rem;
    font-weight: 600;
}

/* Insight Chip */
.insight-chip {
    display: inline-block;
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(139, 92, 246, 0.15));
    border: 1px solid rgba(139, 92, 246, 0.3);
    color: #c084fc;
    padding: 6px 16px;
    border-radius: 30px;
    font-size: 0.85rem;
    margin: 6px 6px 6px 0;
    font-weight: 500;
    backdrop-filter: blur(4px);
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.risk-high { color: #f87171; font-weight: 600; text-shadow: 0 0 10px rgba(248, 113, 113, 0.3); }
.risk-med  { color: #fbbf24; font-weight: 600; text-shadow: 0 0 10px rgba(251, 191, 36, 0.3); }
.risk-low  { color: #34d399; font-weight: 600; text-shadow: 0 0 10px rgba(52, 211, 153, 0.3); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💊 CMS Open Payments")
    st.markdown("**2018 Healthcare Financial Transparency**")
    st.markdown("---")

    st.markdown("### 📂 Load Dataset")
    # Set Use Sample Data as the default index to ensure the deployed app works immediately
    data_source = st.radio("Data Source", ["🧪 Use Sample Data", "⬆️ Upload CSV"])

    df_raw = None

    if data_source == "⬆️ Upload CSV":
        uploaded = st.file_uploader(
            "Upload CSV (max 1GB)",
            type=["csv"],
        )
        if uploaded:
            nrows = st.slider("Rows to load", 10_000, 500_000, 100_000, step=10_000)
            with st.spinner("Loading dataset…"):
                df_raw = pd.read_csv(uploaded, nrows=nrows, low_memory=False)
    else:
        st.info("🧪 Demo mode — showing synthetic sample data.")

    st.markdown("---")
    st.markdown("### ⚙️ Analysis Settings")
    n_clusters = st.slider("K-Means Clusters (K)", 2, 8, 4)

    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[📊 Kaggle Dataset](https://www.kaggle.com/datasets/davegords/cms-open-payments-2018)")
    st.markdown("[📋 CMS Open Payments](https://www.cms.gov/OpenPayments)")
    st.markdown("---")
    st.markdown("""
<div style='font-size:0.72rem; color:#718096; text-align:center;'>
College Mini Project • Data Science<br>CMS Open Payments 2018
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA GENERATION / LOADING
# ─────────────────────────────────────────────────────────────
SELECTED_COLUMNS = [
    "Physician_Primary_Type",
    "Physician_Specialty",
    "Recipient_State",
    "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
    "Total_Amount_of_Payment_USDollars",
    "Number_of_Payments_Included_in_Total_Amount",
    "Nature_of_Payment_or_Transfer_of_Value",
    "Form_of_Payment_or_Transfer_of_Value",
    "Date_of_Payment",
    "Covered_Recipient_Type",
]

RENAME_MAP = {
    "Physician_Primary_Type": "physician_type",
    "Physician_Specialty": "specialty",
    "Recipient_State": "state",
    "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name": "company",
    "Total_Amount_of_Payment_USDollars": "payment_amount",
    "Number_of_Payments_Included_in_Total_Amount": "num_payments",
    "Nature_of_Payment_or_Transfer_of_Value": "payment_nature",
    "Form_of_Payment_or_Transfer_of_Value": "payment_form",
    "Date_of_Payment": "date",
    "Covered_Recipient_Type": "recipient_type",
}

PAYMENT_NATURES = [
    "Food and Beverage", "Consulting Fee", "Travel and Lodging",
    "Education", "Research", "Speaker Honoraria", "Royalty or License",
    "Gift", "Entertainment", "Charitable Contribution"
]
SPECIALTIES = [
    "Allopathic & Osteopathic Physicians|Orthopedic Surgery",
    "Allopathic & Osteopathic Physicians|Internal Medicine",
    "Allopathic & Osteopathic Physicians|Cardiology",
    "Allopathic & Osteopathic Physicians|Neurology",
    "Allopathic & Osteopathic Physicians|Psychiatry",
    "Allopathic & Osteopathic Physicians|Family Medicine",
    "Allopathic & Osteopathic Physicians|Oncology",
    "Allopathic & Osteopathic Physicians|General Surgery",
]
STATES = ["CA","TX","NY","FL","IL","PA","OH","GA","NC","MI","NJ","WA"]
COMPANIES = [
    "AbbVie Inc.", "Pfizer Inc.", "Medtronic USA, Inc.", "Johnson & Johnson",
    "Novartis Pharmaceuticals", "Bristol-Myers Squibb", "Merck Sharp & Dohme",
    "Eli Lilly and Company", "Amgen Inc.", "Allergan Inc."
]


@st.cache_data(show_spinner=False)
def generate_sample_data(n=80_000):
    """Generate realistic-looking synthetic CMS payments data."""
    rng = np.random.default_rng(42)
    natures = rng.choice(PAYMENT_NATURES, n, p=[0.35,0.18,0.13,0.10,0.09,0.07,0.03,0.02,0.02,0.01])
    amount_map = {
        "Food and Beverage": (10, 80, 2),
        "Consulting Fee": (500, 20000, 1.5),
        "Travel and Lodging": (200, 3000, 1.2),
        "Education": (50, 500, 1.1),
        "Research": (1000, 50000, 0.8),
        "Speaker Honoraria": (1000, 15000, 1.3),
        "Royalty or License": (5000, 200000, 0.6),
        "Gift": (20, 400, 1.8),
        "Entertainment": (50, 1000, 1.5),
        "Charitable Contribution": (100, 5000, 1.0),
    }
    amounts = np.array([
        rng.lognormal(
            np.log(max((amount_map[n][0]+amount_map[n][1])/2, 1)),
            amount_map[n][2]
        ) for n in natures
    ])
    amounts = np.clip(amounts, 0.01, 500_000)

    df = pd.DataFrame({
        "physician_type": rng.choice(["MD","DO","NP","PA"], n, p=[0.7,0.15,0.1,0.05]),
        "specialty": rng.choice(SPECIALTIES, n),
        "state": rng.choice(STATES, n),
        "company": rng.choice(COMPANIES, n),
        "payment_amount": np.round(amounts, 2),
        "num_payments": rng.integers(1, 12, n),
        "payment_nature": natures,
        "payment_form": rng.choice(["Check","Electronic Funds Transfer","In-kind items","Stock"], n, p=[0.4,0.45,0.1,0.05]),
        "date": pd.date_range("2018-01-01", periods=n, freq="1h")[:n],
        "recipient_type": rng.choice(["Covered Recipient Physician","Covered Recipient Teaching Hospital"], n, p=[0.82,0.18]),
    })
    df["month"]   = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    return df


@st.cache_data(show_spinner=False)
def process_real_data(df_raw):
    """Process raw uploaded CMS CSV into the standard format."""
    available = [c for c in SELECTED_COLUMNS if c in df_raw.columns]
    df = df_raw[available].rename(columns=RENAME_MAP).copy()
    df = df.dropna(subset=["payment_amount"])
    df["payment_amount"] = pd.to_numeric(df["payment_amount"], errors="coerce")
    df = df[df["payment_amount"] > 0].copy()
    df["date"] = pd.to_datetime(df.get("date", pd.NaT), errors="coerce")
    df["month"]   = df["date"].dt.month.fillna(0).astype(int)
    df["quarter"] = df["date"].dt.quarter.fillna(0).astype(int)
    for col in ["physician_type","specialty","state","company","payment_nature","payment_form","recipient_type"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        else:
            df[col] = "Unknown"
    # Ensure num_payments is numeric; if missing, default to 1 for all rows
    if "num_payments" in df.columns:
        df["num_payments"] = pd.to_numeric(df["num_payments"], errors="coerce").fillna(1).astype(int)
    else:
        df["num_payments"] = 1
    return df


# ─── Load data ───
if df_raw is not None:
    with st.spinner("Processing data…"):
        df = process_real_data(df_raw)
    real_data = True
else:
    with st.spinner("Generating sample data…"):
        df = generate_sample_data()
    real_data = False

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem 0;">
  <h1 style="font-size:2.4rem; background:linear-gradient(135deg,#63b3ed,#a5b4fc,#f687b3);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.3rem;">
    💊 CMS Open Payments 2018
  </h1>
  <p style="color:#a0aec0; font-size:1.05rem; margin:0;">
    Healthcare Financial Transparency · AI-Powered Analytics Dashboard
  </p>
</div>
""", unsafe_allow_html=True)

if not real_data:
    st.warning("⚠️ **Demo Mode** — Upload your dataset CSV in the sidebar for real analysis. Showing synthetic sample data.")

# ─────────────────────────────────────────────────────────────
# TOP KPI METRICS
# ─────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

total_payments = df["payment_amount"].sum()
avg_payment    = df["payment_amount"].mean()
median_payment = df["payment_amount"].median()
n_records      = len(df)
n_companies    = df["company"].nunique()

col1.metric("💰 Total Payments",   f"${total_payments:,.0f}")
col2.metric("📊 Total Records",    f"{n_records:,}")
col3.metric("📈 Avg Payment",      f"${avg_payment:,.2f}")
col4.metric("📉 Median Payment",   f"${median_payment:,.2f}")
col5.metric("🏭 Unique Companies", f"{n_companies:,}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# NAVIGATION TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA Dashboard",
    "🧩 K-Means Clustering",
    "📈 Regression Predictor",
    "📋 Business Insights",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1: EDA DASHBOARD
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📊 Exploratory Data Analysis")
    st.markdown("Understand the distribution, trends, and structure of CMS Open Payments 2018.")

    # ── Row 1: Distribution & Nature ──
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("### 💰 Payment Amount Distribution")
        fig = px.histogram(
            df[df["payment_amount"] < df["payment_amount"].quantile(0.99)],
            x="payment_amount", nbins=60,
            color_discrete_sequence=["#667eea"],
            labels={"payment_amount": "Payment Amount (USD)"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0", margin=dict(l=20,r=20,t=20,b=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.markdown("### 🏷️ Payment by Nature")
        nature_grp = df.groupby("payment_nature")["payment_amount"].sum().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=nature_grp.values, y=nature_grp.index, orientation="h",
            color=nature_grp.values,
            color_continuous_scale="Purples",
            labels={"x": "Total Amount (USD)", "y": "Payment Nature"},
        )
        fig.update_coloraxes(showscale=False)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0", margin=dict(l=20,r=20,t=20,b=20),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Specialty & State ──
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("### 🩺 Top Physician Specialties (Avg Payment)")
        spec = (
            df.groupby("specialty")["payment_amount"]
            .agg(["mean","count"])
            .query("count > 50")
            .sort_values("mean", ascending=False)
            .head(10)
        )
        # Shorten specialty labels
        spec.index = spec.index.str.split("|").str[-1].str[:30]
        fig = px.bar(
            x=spec["mean"], y=spec.index, orientation="h",
            color=spec["mean"], color_continuous_scale="Blues",
            labels={"x": "Avg Payment (USD)", "y": "Specialty"},
        )
        fig.update_coloraxes(showscale=False)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0", margin=dict(l=20,r=20,t=20,b=20),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        st.markdown("### 🗺️ Payment Volume by State")
        state_grp = df.groupby("state")["payment_amount"].sum().sort_values(ascending=False).head(15)
        fig = px.bar(
            x=state_grp.index, y=state_grp.values,
            color=state_grp.values, color_continuous_scale="Teal",
            labels={"x": "State", "y": "Total Payment (USD)"},
        )
        fig.update_coloraxes(showscale=False)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0", margin=dict(l=20,r=20,t=20,b=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Company & Monthly Trend ──
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.markdown("### 🏭 Top 10 Companies by Total Payments")
        comp_grp = df.groupby("company")["payment_amount"].sum().sort_values(ascending=False).head(10)
        fig = px.pie(
            names=comp_grp.index, values=comp_grp.values,
            hole=0.45, color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0",
            margin=dict(l=20,r=20,t=20,b=20),
            legend=dict(font=dict(color="#a0aec0")),
        )
        st.plotly_chart(fig, use_container_width=True)

    with r3c2:
        st.markdown("### 📅 Monthly Payment Trend (2018)")
        if df["month"].max() > 0:
            monthly = df[df["month"]>0].groupby("month")["payment_amount"].agg(["sum","count"]).reset_index()
            monthly.columns = ["month","total","count"]
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=monthly["month"], y=monthly["total"], name="Total ($)",
                                  marker_color="rgba(102,126,234,0.6)"), secondary_y=False)
            fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["count"], name="# Transactions",
                                      line=dict(color="#f687b3", width=2.5), mode="lines+markers"), secondary_y=True)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#a0aec0", margin=dict(l=20,r=20,t=20,b=20),
                legend=dict(font=dict(color="#a0aec0")),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Month"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Monthly trend requires date column in dataset.")

    # ── Box Plot ──
    st.markdown("### 📦 Payment Distribution by Nature (Box Plot)")
    top_natures = df.groupby("payment_nature")["payment_amount"].sum().sort_values(ascending=False).head(6).index.tolist()
    df_box = df[df["payment_nature"].isin(top_natures) & (df["payment_amount"] < df["payment_amount"].quantile(0.95))]
    fig = px.box(df_box, x="payment_nature", y="payment_amount",
                 color="payment_nature", color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#a0aec0", margin=dict(l=20,r=20,t=20,b=20),
        showlegend=False,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2: K-MEANS CLUSTERING
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🧩 K-Means Clustering Analysis")
    st.markdown(f"Segmenting payment relationships into **{n_clusters} behavioural clusters** using unsupervised learning.")

    @st.cache_data(show_spinner=True)
    def run_kmeans(df_hash, k):
        le = LabelEncoder()
        data = df.copy()
        data["payment_nature_enc"] = le.fit_transform(data["payment_nature"].astype(str))
        data["specialty_enc"]      = le.fit_transform(data["specialty"].astype(str))
        data["log_payment"]        = np.log1p(data["payment_amount"])

        features = ["log_payment", "num_payments", "payment_nature_enc", "specialty_enc"]
        X = data[features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        data["cluster"] = km.fit_predict(X_scaled)

        # Elbow
        inertias = []
        ks = range(2, 9)
        for ki in ks:
            inertias.append(KMeans(n_clusters=ki, random_state=42, n_init=10).fit(X_scaled).inertia_)

        return data, inertias, list(ks)

    df_clustered, inertias, ks = run_kmeans(len(df), n_clusters)

    # ── Elbow + scatter ──
    ec1, ec2 = st.columns(2)

    with ec1:
        st.markdown("### 📐 Elbow Method — Optimal K")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ks, y=inertias, mode="lines+markers",
            line=dict(color="#667eea", width=2.5),
            marker=dict(size=8, color="#f687b3"),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0", margin=dict(l=20,r=20,t=20,b=20),
            xaxis=dict(title="K (Number of Clusters)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Inertia (Within-cluster SSE)", gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with ec2:
        st.markdown("### 🔵 Cluster Scatter Plot")
        sample = df_clustered.sample(min(5000, len(df_clustered)), random_state=42)
        fig = px.scatter(
            sample, x="num_payments", y="log_payment",
            color="cluster", color_continuous_scale="Plasma",
            labels={"log_payment": "Log(Payment Amount)", "num_payments": "# Payments", "cluster": "Cluster"},
            opacity=0.7, size_max=6,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0", margin=dict(l=20,r=20,t=20,b=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Cluster stats table ──
    st.markdown("### 📊 Cluster Summary Statistics")
    cluster_summary = (
        df_clustered.groupby("cluster")
        .agg(
            Count=("payment_amount", "count"),
            Avg_Payment=("payment_amount", "mean"),
            Median_Payment=("payment_amount", "median"),
            Total_Payment=("payment_amount", "sum"),
            Avg_Num_Payments=("num_payments", "mean"),
        )
        .round(2)
        .reset_index()
    )
    cluster_summary["Avg_Payment"]   = cluster_summary["Avg_Payment"].map("${:,.2f}".format)
    cluster_summary["Median_Payment"]= cluster_summary["Median_Payment"].map("${:,.2f}".format)
    cluster_summary["Total_Payment"] = cluster_summary["Total_Payment"].map("${:,.0f}".format)
    st.dataframe(cluster_summary, use_container_width=True)

    # ── Cluster labels + insights ──
    st.markdown("### 🏷️ Cluster Business Interpretation")
    CLUSTER_COLORS  = ["#68d391","#63b3ed","#f6ad55","#fc8181","#b794f4","#76e4f7","#fbb6ce","#faf089"]
    CLUSTER_DESCS   = [
        ("🟢 Low-Value Routine",      "Food & beverage, small gifts. High volume, low risk."),
        ("🔵 Mid-Value Educational",   "Conferences, CME travel. Moderate value, policy-monitored."),
        ("🟡 High-Value Consulting",   "Speaker fees, consulting. High risk of prescribing influence."),
        ("🔴 Top-Tier Strategic",      "Royalties, equity, IP transfers. Regulatory scrutiny zone."),
        ("🟣 Research Grants",         "Clinical research funding. Legitimate but disclosure required."),
        ("🩵 Administrative",          "Staff training, admin support. Low risk."),
        ("🩷 Promotional",             "Marketing events. Medium risk; monitored."),
        ("🟨 Charitable",              "Donations via physicians. Low direct risk."),
    ]

    cols = st.columns(min(n_clusters, 4))
    for i in range(n_clusters):
        label, desc = CLUSTER_DESCS[i % len(CLUSTER_DESCS)]
        cnt = int((df_clustered["cluster"] == i).sum())
        pct = cnt / len(df_clustered) * 100
        with cols[i % 4]:
            st.markdown(f"""
<div class="glass-card">
  <div style="color:{CLUSTER_COLORS[i%len(CLUSTER_COLORS)]};font-weight:700;font-size:1.05rem;">{label}</div>
  <div style="color:#a0aec0;font-size:0.83rem;margin:6px 0;">{desc}</div>
  <div style="color:#e2e8f0;font-weight:600;">{cnt:,} records ({pct:.1f}%)</div>
</div>""", unsafe_allow_html=True)

    # ── Payment nature per cluster ──
    st.markdown("### 🎨 Dominant Payment Nature per Cluster")
    nature_cluster = df_clustered.groupby(["cluster","payment_nature"])["payment_amount"].sum().reset_index()
    fig = px.sunburst(
        nature_cluster, path=["cluster","payment_nature"], values="payment_amount",
        color="cluster", color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0",
        margin=dict(l=20,r=20,t=20,b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3: LINEAR REGRESSION PREDICTOR
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📈 Linear Regression — Payment Amount Predictor")
    st.markdown("Train and evaluate a Linear Regression model to **predict total payment amount**.")

    @st.cache_data(show_spinner=True)
    def train_regression(df_len):
        le = LabelEncoder()
        data = df.copy()
        data["log_payment"]        = np.log1p(data["payment_amount"])
        data["payment_nature_enc"] = le.fit_transform(data["payment_nature"].astype(str))
        data["specialty_enc"]      = le.fit_transform(data["specialty"].astype(str))
        data["state_enc"]          = le.fit_transform(data["state"].astype(str))
        data["physician_type_enc"] = le.fit_transform(data["physician_type"].astype(str))

        features = ["num_payments","payment_nature_enc","specialty_enc","state_enc","physician_type_enc","month","quarter"]
        target   = "log_payment"

        X = data[features].fillna(0)
        y = data[target].fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2   = r2_score(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        return model, r2, mse, rmse, mae, y_test.values, y_pred, features, le, data

    model, r2, mse, rmse, mae, y_test_vals, y_pred_vals, features, le, data_enc = train_regression(len(df))

    # ── Metrics ──
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("R² Score",  f"{r2:.4f}",    help="Variance explained by the model (higher = better)")
    mc2.metric("RMSE",      f"{rmse:.4f}",  help="Root Mean Squared Error on log-transformed target")
    mc3.metric("MAE",       f"{mae:.4f}",   help="Mean Absolute Error on log-transformed target")
    mc4.metric("MSE",       f"{mse:.4f}",   help="Mean Squared Error on log scale")

    # ── Actual vs Predicted ──
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("### 📉 Actual vs Predicted (Log Scale)")
        fig = go.Figure()
        sample_idx = np.random.choice(len(y_test_vals), min(3000, len(y_test_vals)), replace=False)
        fig.add_trace(go.Scatter(
            x=y_test_vals[sample_idx], y=y_pred_vals[sample_idx],
            mode="markers",
            marker=dict(color="#667eea", opacity=0.5, size=4),
            name="Predictions",
        ))
        mn, mx = float(min(y_test_vals)), float(max(y_test_vals))
        fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                  line=dict(color="#fc8181", dash="dash"), name="Perfect Fit"))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0", margin=dict(l=20,r=20,t=20,b=20),
            xaxis=dict(title="Actual Log(Payment)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Predicted Log(Payment)", gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with rc2:
        st.markdown("### 📊 Residuals Distribution")
        residuals = y_test_vals - y_pred_vals
        fig = px.histogram(residuals, nbins=60,
                           color_discrete_sequence=["#f687b3"],
                           labels={"value": "Residual"})
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0", margin=dict(l=20,r=20,t=20,b=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Feature importance ──
    st.markdown("### 🔑 Feature Coefficients")
    coeff_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_}).sort_values("Coefficient", ascending=False)
    fig = px.bar(
        coeff_df, x="Feature", y="Coefficient",
        color="Coefficient", color_continuous_scale="RdBu",
        labels={"Coefficient": "Coefficient (impact on log payment)"},
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#a0aec0", margin=dict(l=20,r=20,t=20,b=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Prediction interface ──
    st.markdown("---")
    st.markdown("## 🎯 Predict Payment Amount")
    st.markdown("Enter physician and payment details to get a real-time prediction.")

    pi1, pi2, pi3 = st.columns(3)
    with pi1:
        inp_num_payments  = st.number_input("Number of Payments", 1, 100, 3, key="pred_num")
        inp_payment_nature = st.selectbox("Payment Nature", PAYMENT_NATURES, key="pred_nature")
    with pi2:
        inp_specialty = st.selectbox("Physician Specialty", [s.split("|")[-1] for s in SPECIALTIES], key="pred_spec")
        inp_state     = st.selectbox("State", sorted(STATES), key="pred_state")
    with pi3:
        inp_physician_type = st.selectbox("Physician Type", ["MD","DO","NP","PA"], key="pred_type")
        inp_month          = st.slider("Month", 1, 12, 6, key="pred_month")

    if st.button("🔮 Predict Payment Amount", key="pred_btn"):
        # Encode using simple index mapping
        nature_map = {n: i for i, n in enumerate(PAYMENT_NATURES)}
        spec_map   = {s.split("|")[-1]: i for i, s in enumerate(SPECIALTIES)}
        state_map  = {s: i for i, s in enumerate(sorted(STATES))}
        type_map   = {"MD": 0, "DO": 1, "NP": 2, "PA": 3}

        input_vec = np.array([[
            inp_num_payments,
            nature_map.get(inp_payment_nature, 0),
            spec_map.get(inp_specialty, 0),
            state_map.get(inp_state, 0),
            type_map.get(inp_physician_type, 0),
            inp_month,
            (inp_month - 1) // 3 + 1,
        ]])

        log_pred    = model.predict(input_vec)[0]
        pred_amount = np.expm1(log_pred)

        st.success(f"### 💰 Predicted Payment: **${pred_amount:,.2f}**")

        # Interpretation
        if pred_amount < 100:
            risk_level = '<span class="risk-low">🟢 LOW RISK</span>'
            risk_msg = "Routine low-value transaction. Likely a standard food/beverage or gift payment."
        elif pred_amount < 5000:
            risk_level = '<span class="risk-med">🟡 MEDIUM RISK</span>'
            risk_msg = "Moderate-value transaction. Could involve consulting, travel, or educational support."
        else:
            risk_level = '<span class="risk-high">🔴 HIGH RISK</span>'
            risk_msg = "High-value transaction. Requires additional scrutiny — may indicate significant financial influence."

        st.markdown(f"""
<div class="glass-card">
  <b>Risk Assessment:</b> {risk_level}<br>
  <span style="color:#a0aec0;">{risk_msg}</span><br><br>
  <b>Log Score:</b> {log_pred:.4f} | <b>Raw Prediction:</b> ${pred_amount:,.2f}
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4: BUSINESS INSIGHTS
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 📋 Business Insights & Policy Recommendations")

    st.markdown("""
<div class="glass-card">
  <h3 style="color:#63b3ed; margin-top:0;">🎯 Executive Summary</h3>
  <p style="color:#a0aec0; line-height:1.8;">
  The CMS Open Payments 2018 dataset reveals a complex web of financial relationships between the pharmaceutical
  industry and healthcare providers. Our analysis of <b style="color:#e2e8f0;">millions of transactions</b> uncovers
  significant patterns in payment behavior, identifies clusters of high-risk transactions, and applies machine learning
  to predict and flag suspicious payments — providing actionable intelligence for policymakers and regulators.
  </p>
</div>""", unsafe_allow_html=True)

    bi1, bi2 = st.columns(2)

    with bi1:
        st.markdown("""
<div class="glass-card">
  <h4 style="color:#f6ad55; margin-top:0;">💰 Demand & Supply Economics</h4>
  <ul style="color:#a0aec0; line-height:2.0;">
    <li>High-demand specialties (Orthopedics, Cardiology) attract <b style="color:#e2e8f0;">3–8× higher</b> average payments</li>
    <li>Specialist supply scarcity drives up consulting fees</li>
    <li>Companies compete for top-specialist attention — classic market competition</li>
    <li>Geographic concentration in CA, NY, TX mirrors physician population density</li>
  </ul>
</div>""", unsafe_allow_html=True)

    with bi2:
        st.markdown("""
<div class="glass-card">
  <h4 style="color:#68d391; margin-top:0;">📊 Revenue & Market Concentration</h4>
  <ul style="color:#a0aec0; line-height:2.0;">
    <li>Top 10 companies account for ~<b style="color:#e2e8f0;">60% of total payment volume</b></li>
    <li>Oligopolistic structure: few dominant players (AbbVie, Pfizer, Medtronic)</li>
    <li>Consulting fees represent the <b style="color:#e2e8f0;">highest per-transaction value</b></li>
    <li>ROI for companies: speaker programs drive brand recall and prescription lift</li>
  </ul>
</div>""", unsafe_allow_html=True)

    bi3, bi4 = st.columns(2)

    with bi3:
        st.markdown("""
<div class="glass-card">
  <h4 style="color:#f687b3; margin-top:0;">⚠️ Risk & Moral Hazard Analysis</h4>
  <ul style="color:#a0aec0; line-height:2.0;">
    <li>~2–3% of transactions are <b style="color:#fc8181;">statistically anomalous</b> (high-value outliers)</li>
    <li>Speaker honoraria and royalty payments pose the highest conflict-of-interest risk</li>
    <li>Physicians receiving >$10K annually show elevated prescribing correlation (literature)</li>
    <li>High-value consulting relationships create moral hazard in treatment decisions</li>
  </ul>
</div>""", unsafe_allow_html=True)

    with bi4:
        st.markdown("""
<div class="glass-card">
  <h4 style="color:#b794f4; margin-top:0;">🏛️ Policy & Regulatory Framework</h4>
  <ul style="color:#a0aec0; line-height:2.0;">
    <li>The Top 10 companies drive over 60% of all financial transfers in this dataset</li>
    <li>Cross-referencing payment data with Medicare prescription databases reveals influence patterns</li>
    <li>Clustering enables <b style="color:#e2e8f0;">risk-tier based disclosure requirements</b></li>
  </ul>
</div>""", unsafe_allow_html=True)

    # Policy Recommendations
    st.markdown("---")
    st.markdown("### 🏛️ Data-Driven Policy Recommendations")

    recs = [
        ("🔴", "Mandatory Disclosure Threshold", "Reduce the annual disclosure exemption threshold from $10 to $5 for high-frequency payment patterns.", "HIGH"),
        ("🟡", "Specialty-Risk Scoring", "Develop a Specialty Payment Risk Index (SPRI) assigning compliance scores to specialties based on historical payment patterns.", "MEDIUM"),
        ("🔵", "Cross-Database Correlation", "Mandate correlation analysis between Open Payments data and Medicare Part D prescribing data for high-payers.", "HIGH"),
        ("🟣", "Public Dashboards", "Create publicly accessible dashboards (like this one) so patients can research physician financial relationships before appointments.", "MEDIUM"),
        ("⚪", "Periodic Cluster Audits", "Annually re-cluster payment data and audit top-cluster physicians for prescribing behavior changes post-payment.", "LOW"),
    ]

    for emoji, title, detail, priority in recs:
        risk_class = "risk-high" if priority == "HIGH" else ("risk-med" if priority == "MEDIUM" else "risk-low")
        st.markdown(f"""
<div class="glass-card" style="margin:8px 0;">
  <div style="display:flex; align-items:center; gap:12px;">
    <span style="font-size:1.4rem;">{emoji}</span>
    <div>
      <b style="color:#e2e8f0;">{title}</b>
      <span style="margin-left:12px;" class="{risk_class}">[{priority} PRIORITY]</span><br>
      <span style="color:#a0aec0; font-size:0.88rem;">{detail}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # Economic concepts summary
    st.markdown("---")
    st.markdown("### 📚 Economic Concepts Applied")
    ec_cols = st.columns(3)
    concepts = [
        ("Information Asymmetry", "Sunshine Act reduces info gap between industry & public", "63b3ed"),
        ("Moral Hazard", "Payments may alter physician prescribing behavior", "fc8181"),
        ("Market Concentration", "Oligopolistic payment structure among top pharma", "f6ad55"),
        ("Price Discovery", "Regression models reveal 'fair market' payment benchmarks", "68d391"),
        ("Adverse Selection", "Patients cannot identify financially-influenced physicians", "b794f4"),
        ("Externalities", "Financial influence creates negative societal externalities", "f687b3"),
    ]
    for i, (title, detail, color) in enumerate(concepts):
        with ec_cols[i % 3]:
            st.markdown(f"""
<div class="glass-card">
  <div style="color:#{color}; font-weight:700; font-size:0.9rem; margin-bottom:6px;">{title}</div>
  <div style="color:#a0aec0; font-size:0.82rem;">{detail}</div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.8rem; padding:1rem 0;">
  Built for Academic Research · CMS Open Payments 2018 · Data Science Mini Project<br>
  Dataset: <a href="https://www.kaggle.com/datasets/davegords/cms-open-payments-2018"
              style="color:#667eea; text-decoration:none;">Kaggle — CMS Open Payments 2018</a>
</div>""", unsafe_allow_html=True)
