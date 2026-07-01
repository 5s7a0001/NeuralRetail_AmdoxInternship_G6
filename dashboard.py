# ============================================================
# NeuralRetail — Streamlit Dashboard
# Amdox Technologies | Data Science & Analytics
# Group 6 — Amdox Internship
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION  (must be very first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="NeuralRetail | Amdox Technologies",
    page_icon="\U0001f6cd",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Amdox brand dark theme
# ============================================================
st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap");
html, body, [class*="css"]        { font-family: "Inter", sans-serif; }
.stApp, .main                     { background-color: #0a0a0f; }
[data-testid="stSidebar"]         {
    background: linear-gradient(180deg, #0f0f18 0%, #1a1a28 100%);
    border-right: 1px solid rgba(232,78,27,0.3);
}
/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, #141420 0%, #1e1e30 100%);
    border: 1px solid rgba(232,78,27,0.35);
    border-radius: 12px;
    padding: 20px 14px;
    text-align: center;
    transition: transform .18s ease, box-shadow .18s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(232,78,27,.22);
}
.kpi-icon  { font-size: 26px; }
.kpi-value { font-size: 26px; font-weight: 700; color: #E84E1B; margin-top: 4px; }
.kpi-label { font-size: 11px; color: #777; margin-top: 5px;
             text-transform: uppercase; letter-spacing: .9px; }
/* Typography */
h1 { color: #E84E1B !important; font-weight: 700; }
h2 { color: #F7941D !important; font-weight: 600; }
h3 { color: #FBBA13 !important; font-weight: 600; }
[data-testid="stMetricValue"] { color: #E84E1B; font-size: 22px; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #999; font-size: 12px; }
hr { border-color: rgba(232,78,27,.2); margin: 16px 0; }
/* Sidebar date picker label */
.sidebar-section {
    background: rgba(232,78,27,.08);
    border: 1px solid rgba(232,78,27,.25);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 10px;
}
/* Tabs */
.stTabs [data-baseweb="tab-list"] { background-color: #111118; border-radius: 8px; }
.stTabs [aria-selected="true"]    { color: #E84E1B !important;
                                    border-bottom: 2px solid #E84E1B; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
PLOT_LAY = dict(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cccccc", family="Inter"),
    margin=dict(t=42, b=22, l=10, r=10),
    legend=dict(bgcolor="rgba(0,0,0,0)")
)
COLORS = ["#E84E1B","#F7941D","#FBBA13","#27AE60","#2980B9","#8E44AD","#16A085","#C0392B"]
HEAT   = ["#FBBA13","#F7941D","#E84E1B"]

BASE        = os.path.dirname(os.path.abspath(__file__))
DATA        = os.path.join(BASE, "data")
MODELS_PATH = os.path.join(BASE, "models")


# ============================================================
# DATA LOADING (cached)
# ============================================================
@st.cache_data
def load_data():
    rfm       = pd.read_csv(os.path.join(DATA, "rfm_features.csv"))
    daily     = pd.read_csv(os.path.join(DATA, "daily_sales.csv"), parse_dates=["Date"])
    churn     = pd.read_csv(os.path.join(DATA, "churn_predictions.csv"))
    segments  = pd.read_csv(os.path.join(DATA, "customer_segments.csv"))
    inventory = pd.read_csv(os.path.join(DATA, "inventory_plan.csv"))
    return rfm, daily, churn, segments, inventory


@st.cache_resource
def load_models():
    try:
        churn_model    = joblib.load(os.path.join(MODELS_PATH, "churn_model.pkl"))
        # churn_features.pkl contains the TRAINING DataFrame — extract column names only
        _raw           = joblib.load(os.path.join(MODELS_PATH, "churn_features.pkl"))
        if isinstance(_raw, pd.DataFrame):
            churn_feat_cols = list(_raw.columns)
        elif isinstance(_raw, (list, np.ndarray)):
            churn_feat_cols = list(_raw)
        else:
            churn_feat_cols = None

        seg_model      = joblib.load(os.path.join(MODELS_PATH, "customer_segmentation_model.pkl"))
        seg_scaler     = joblib.load(os.path.join(MODELS_PATH, "customer_segmentation_scaler.pkl"))
        forecast_model = joblib.load(os.path.join(MODELS_PATH, "lightgbm_forecasting_model.pkl"))
        return churn_model, churn_feat_cols, seg_model, seg_scaler, forecast_model
    except Exception as e:
        st.warning(f"\u26a0\ufe0f Model loading issue: {e}")
        return None, None, None, None, None


with st.spinner("\U0001f504 Loading NeuralRetail data..."):
    rfm, daily, churn_df, segments, inventory = load_data()
    churn_model, churn_feat_cols, seg_model, seg_scaler, forecast_model = load_models()


# ============================================================
# HELPER — KPI card renderer
# ============================================================
def kpi(col, icon, label, value):
    col.markdown(
        f'''<div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>''',
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR — logo, navigation, date filter
# ============================================================
with st.sidebar:
    # Brand logo
    st.markdown("""
    <div style="text-align:center;padding:18px 0 10px;">
      <div style="background:linear-gradient(135deg,#E84E1B,#F7941D);
                  border-radius:10px;padding:12px 24px;display:inline-block;
                  box-shadow:0 4px 18px rgba(232,78,27,.4);">
        <span style="color:#fff;font-size:22px;font-weight:800;letter-spacing:2.5px;">AMDOX</span>
      </div>
      <div style="color:#555;font-size:10px;margin-top:6px;letter-spacing:1.5px;">TECHNOLOGIES</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### \U0001f6cd NeuralRetail")
    st.caption("**AI Sales Intelligence Platform** | Group 6")
    st.markdown("---")

    # Page navigation
    PAGE_LABELS = [
        "\U0001f4ca Executive Overview",
        "\U0001f4c8 Sales & Demand Forecast",
        "\U0001f465 Customer Intelligence",
        "\u26a0\ufe0f  Churn Risk Analysis",
        "\U0001f4e6 Inventory Optimization",
        "\U0001f9ea Model Performance",
    ]
    page = st.radio("", PAGE_LABELS, label_visibility="collapsed")

    st.markdown("---")

    # ── Date range filter (applies to daily-based pages) ──────
    st.markdown("**\U0001f4c5 Date Range Filter**")
    d_min_all = daily["Date"].min().date()
    d_max_all = daily["Date"].max().date()

    date_from = st.date_input("From", value=d_min_all,
                              min_value=d_min_all, max_value=d_max_all,
                              key="date_from")
    date_to   = st.date_input("To",   value=d_max_all,
                              min_value=d_min_all, max_value=d_max_all,
                              key="date_to")

    if date_from > date_to:
        st.error("'From' must be before 'To'")
        date_from, date_to = d_min_all, d_max_all

    # Filtered daily dataframe (used across pages)
    mask         = (daily["Date"].dt.date >= date_from) & (daily["Date"].dt.date <= date_to)
    daily_filt   = daily[mask].copy()

    st.markdown("---")

    # Quick stats
    st.caption(f"\U0001f464 {rfm.shape[0]:,} customers")
    st.caption(f"\U0001f4e6 {inventory.shape[0]:,} SKU rows  ({inventory['StockCode'].nunique():,} unique)")
    st.caption(f"\U0001f4c5 {daily_filt.shape[0]:,} days in view")
    st.markdown("---")
    st.caption("Built with \u2764\ufe0f by **Amdox DS Team** | 2026")


# ── Apply date range info label ───────────────────────────────
date_label = f"{date_from} \u2192 {date_to}"


# ============================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ============================================================
if page == PAGE_LABELS[0]:
    st.title("\U0001f4ca Executive Overview")
    st.markdown(f"**Key Performance Indicators** | {date_label}")
    st.markdown("---")

    dc = daily_filt.dropna(subset=["Revenue"])

    total_revenue     = dc["Revenue"].sum()
    total_orders      = dc["Orders"].sum()
    total_quantity    = dc["Quantity"].sum()
    total_customers   = rfm["CustomerID"].nunique()
    avg_daily_revenue = dc["Revenue"].mean() if len(dc) > 0 else 0
    total_sku_rows    = inventory.shape[0]
    total_sku_unique  = inventory["StockCode"].nunique()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpi(c1,"\U0001f4b0","Total Revenue",   f"\u00a3{total_revenue:,.0f}")
    kpi(c2,"\U0001f6d2","Total Orders",    f"{int(total_orders):,}")
    kpi(c3,"\U0001f4e6","Qty Sold",        f"{int(total_quantity):,}")
    kpi(c4,"\U0001f465","Customers",       f"{total_customers:,}")
    kpi(c5,"\U0001f4c8","Avg Daily Rev",   f"\u00a3{avg_daily_revenue:,.0f}")
    kpi(c6,"\U0001f3f7\ufe0f","SKU Rows / Unique",f"{total_sku_rows:,} / {total_sku_unique:,}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("\U0001f4c8 Daily Revenue Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dc["Date"], y=dc["Revenue"],
            name="Daily Revenue", line=dict(color="#E84E1B", width=1.5),
            fill="tozeroy", fillcolor="rgba(232,78,27,0.07)"
        ))
        for col_name, color in [("Rolling_Mean_7","#F7941D"),("Rolling_Mean_30","#FBBA13")]:
            if col_name in dc.columns:
                fig.add_trace(go.Scatter(x=dc["Date"], y=dc[col_name],
                                         name=col_name.replace("_"," "),
                                         line=dict(color=color, width=2)))
        fig.update_layout(**PLOT_LAY, hovermode="x unified",
                          xaxis_title="Date", yaxis_title="Revenue (\u00a3)")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("\U0001f4c5 Monthly Revenue")
        dc2 = dc.copy()
        dc2["Month"] = dc2["Date"].dt.to_period("M").astype(str)
        monthly = dc2.groupby("Month")["Revenue"].sum().reset_index()
        fig2 = px.bar(monthly, x="Month", y="Revenue",
                      color="Revenue", color_continuous_scale=HEAT,
                      labels={"Revenue":"Revenue (\u00a3)","Month":""})
        fig2.update_layout(**PLOT_LAY, xaxis_tickangle=-45, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("\U0001f6d2 Orders vs Revenue")
        fig3 = px.scatter(
            dc.dropna(subset=["Orders","Revenue"]),
            x="Orders", y="Revenue", color="Revenue",
            color_continuous_scale=HEAT, opacity=0.7, trendline="ols",
            labels={"Orders":"Daily Orders","Revenue":"Revenue (\u00a3)"}
        )
        fig3.update_layout(**PLOT_LAY, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("\U0001f4c6 Revenue by Day of Week")
        dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
        dc3 = dc.copy()
        dc3["DOW"] = dc3["DayOfWeek"].map(dow_map)
        dow_rev = dc3.groupby("DOW")["Revenue"].mean().reset_index()
        dow_order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        dow_rev["DOW"] = pd.Categorical(dow_rev["DOW"], categories=dow_order, ordered=True)
        dow_rev = dow_rev.sort_values("DOW")
        fig4 = px.bar(dow_rev, x="DOW", y="Revenue",
                      color="Revenue", color_continuous_scale=HEAT,
                      labels={"Revenue":"Avg Revenue (\u00a3)","DOW":"Day"})
        fig4.update_layout(**PLOT_LAY, coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    # Revenue heatmap
    st.subheader("\U0001f321\ufe0f Revenue Heatmap — Quarter × Month")
    try:
        dc4 = dc.copy()
        dc4["Qtr"]     = dc4["Date"].dt.quarter.map({1:"Q1",2:"Q2",3:"Q3",4:"Q4"})
        dc4["MonthNo"] = dc4["Date"].dt.month
        heat = dc4.groupby(["Qtr","MonthNo"])["Revenue"].sum().reset_index()
        piv  = heat.pivot(index="Qtr", columns="MonthNo", values="Revenue").fillna(0)
        piv.columns = [f"Month {c}" for c in piv.columns]
        fh = px.imshow(piv, color_continuous_scale=HEAT,
                       labels=dict(color="Revenue (\u00a3)"))
        fh.update_layout(**PLOT_LAY)
        st.plotly_chart(fh, use_container_width=True)
    except Exception:
        st.info("Not enough data in the selected range for the heatmap.")


# ============================================================
# PAGE 2 — SALES & DEMAND FORECAST
# ============================================================
elif page == PAGE_LABELS[1]:
    st.title("\U0001f4c8 Sales & Demand Forecast")
    st.markdown(f"**Time-Series Analysis | LightGBM Model** | {date_label}")
    st.markdown("---")

    dc = daily_filt.dropna(subset=["Date","Revenue"])

    # Model banner
    if forecast_model is not None:
        st.success("\u2705 LightGBM Forecasting Model loaded")
    else:
        st.warning("\u26a0\ufe0f Forecasting model not loaded — showing historical analysis")

    st.subheader("\U0001f52e Revenue + Rolling Averages")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dc["Date"], y=dc["Revenue"],
                             name="Daily Revenue", line=dict(color="#E84E1B",width=1.5),
                             fill="tozeroy", fillcolor="rgba(232,78,27,0.07)"))
    for col_n, color in [("Rolling_Mean_7","#F7941D"),("Rolling_Mean_30","#FBBA13")]:
        if col_n in dc.columns:
            fig.add_trace(go.Scatter(x=dc["Date"], y=dc[col_n], name=col_n.replace("_"," "),
                                     line=dict(color=color,width=2)))
    fig.update_layout(**PLOT_LAY, hovermode="x unified",
                      xaxis_title="Date", yaxis_title="Revenue (\u00a3)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("\u23f3 Revenue Lag Comparison")
        lag_cols = [c for c in ["Revenue_Lag_1","Revenue_Lag_7","Revenue_Lag_14","Revenue_Lag_30"]
                    if c in dc.columns]
        if lag_cols:
            lf = go.Figure()
            for lag, clr in zip(lag_cols, ["#E84E1B","#F7941D","#FBBA13","#27AE60"]):
                lf.add_trace(go.Scatter(x=dc["Date"], y=dc[lag],
                                        name=lag.replace("_"," "),
                                        line=dict(color=clr,width=1.5)))
            lf.update_layout(**PLOT_LAY, hovermode="x unified",
                             xaxis_title="Date", yaxis_title="Revenue (\u00a3)")
            st.plotly_chart(lf, use_container_width=True)

    with col2:
        st.subheader("\U0001f4ca Revenue Distribution")
        fh = px.histogram(dc, x="Revenue", nbins=40,
                          color_discrete_sequence=["#E84E1B"],
                          labels={"Revenue":"Daily Revenue (\u00a3)"})
        fh.update_layout(**PLOT_LAY)
        st.plotly_chart(fh, use_container_width=True)

    st.markdown("---")

    st.subheader("\U0001f9e0 LightGBM Feature Importance")
    if forecast_model is not None:
        try:
            feat_names = ["Revenue_Lag_1","Revenue_Lag_7","Revenue_Lag_14","Revenue_Lag_30",
                          "Rolling_Mean_7","Rolling_Mean_30","Rolling_Std_7",
                          "DayOfWeek","Month","Quarter","IsWeekend"]
            imps = forecast_model.feature_importances_
            n    = min(len(feat_names), len(imps))
            fi   = pd.DataFrame({"Feature":feat_names[:n],"Importance":imps[:n]})                     .sort_values("Importance", ascending=True)
            ff = px.bar(fi, x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale=HEAT)
            ff.update_layout(**PLOT_LAY, coloraxis_showscale=False)
            st.plotly_chart(ff, use_container_width=True)
        except Exception:
            st.info("Feature importance unavailable for this model version.")
    else:
        st.info("Load the LightGBM model to view feature importance.")

    st.markdown("---")

    st.subheader("\U0001f4d0 Forecast Performance Targets")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("\U0001f3af Model",          "LightGBM")
    c2.metric("\U0001f4ca Target MAPE",    "\u2264 8%")
    c3.metric("\U0001f4c8 R\u00b2 Target","\u2265 0.90")
    c4.metric("\u23f1\ufe0f Granularity", "Daily")


# ============================================================
# PAGE 3 — CUSTOMER INTELLIGENCE
# ============================================================
elif page == PAGE_LABELS[2]:
    st.title("\U0001f465 Customer Intelligence")
    st.markdown("**RFM Segmentation — Know Your Customers**")
    st.markdown("---")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("\U0001f464 Customers",     f"{rfm['CustomerID'].nunique():,}")
    c2.metric("\u23f0 Avg Recency",       f"{rfm['Recency'].mean():.0f} days")
    c3.metric("\U0001f501 Avg Frequency", f"{rfm['Frequency'].mean():.1f}")
    c4.metric("\U0001f4b0 Avg Monetary",  f"\u00a3{rfm['Monetary'].mean():,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("\U0001f967 Segment Distribution")
        seg_c = rfm["Segment"].value_counts().reset_index()
        seg_c.columns = ["Segment","Count"]
        fp = px.pie(seg_c, values="Count", names="Segment",
                    color_discrete_sequence=COLORS, hole=0.45)
        fp.update_layout(**PLOT_LAY)
        fp.update_traces(textposition="inside", textinfo="percent+label", textfont_size=11)
        st.plotly_chart(fp, use_container_width=True)

    with col2:
        st.subheader("\U0001f4b0 Revenue by Segment")
        sr = rfm.groupby("Segment")["Monetary"].sum().reset_index()
        sr = sr.sort_values("Monetary", ascending=False)
        fs = px.bar(sr, x="Segment", y="Monetary",
                    color="Monetary", color_continuous_scale=HEAT,
                    labels={"Monetary":"Total Revenue (\u00a3)","Segment":""})
        fs.update_layout(**PLOT_LAY, coloraxis_showscale=False, xaxis_tickangle=-30)
        st.plotly_chart(fs, use_container_width=True)

    st.markdown("---")

    st.subheader("\U0001f4c8 RFM Distributions")
    c3,c4,c5 = st.columns(3)
    with c3:
        f3 = px.histogram(rfm, x="Recency", nbins=30, title="Recency (days)",
                          color_discrete_sequence=["#E84E1B"])
        f3.update_layout(**PLOT_LAY)
        st.plotly_chart(f3, use_container_width=True)
    with c4:
        f4 = px.histogram(rfm, x="Frequency", nbins=30, title="Purchase Frequency",
                          color_discrete_sequence=["#F7941D"])
        f4.update_layout(**PLOT_LAY)
        st.plotly_chart(f4, use_container_width=True)
    with c5:
        f5 = px.histogram(rfm, x="Monetary", nbins=30, title="Monetary Value (\u00a3)",
                          color_discrete_sequence=["#FBBA13"])
        f5.update_layout(**PLOT_LAY)
        st.plotly_chart(f5, use_container_width=True)

    st.markdown("---")

    st.subheader("\U0001f310 3D RFM Scatter — Customer Clusters")
    sample = rfm.sample(min(1000, len(rfm)), random_state=42)
    f3d = px.scatter_3d(sample, x="Recency", y="Frequency", z="Monetary",
                        color="Segment", color_discrete_sequence=COLORS, opacity=0.7,
                        labels={"Recency":"Recency (days)","Frequency":"Orders",
                                "Monetary":"Revenue (\u00a3)"})
    f3d.update_layout(**PLOT_LAY)
    st.plotly_chart(f3d, use_container_width=True)

    st.markdown("---")

    st.subheader("\U0001f50d Customer Details")
    cf1, cf2 = st.columns([2,1])
    with cf1:
        seg_sel = st.selectbox("Filter by Segment:",
                               ["All"] + sorted(rfm["Segment"].dropna().unique().tolist()))
    with cf2:
        top_n = st.slider("Show top N customers:", 10, 500, 50, step=10)

    disp = rfm if seg_sel == "All" else rfm[rfm["Segment"] == seg_sel]
    dcols = [c for c in ["CustomerID","Recency","Frequency","Monetary",
                          "R_Score","F_Score","M_Score","RFM_Score","Segment","Churned"]
             if c in disp.columns]
    st.dataframe(
        disp[dcols].sort_values("Monetary", ascending=False).head(top_n).reset_index(drop=True),
        use_container_width=True, height=400
    )

    if "Customer_Segment" in segments.columns:
        st.markdown("---")
        st.subheader("\U0001f4ca K-Means Cluster Distribution")
        cc = segments["Customer_Segment"].value_counts().reset_index()
        cc.columns = ["Cluster","Count"]
        fc = px.bar(cc, x="Cluster", y="Count",
                    color="Count", color_continuous_scale=HEAT)
        fc.update_layout(**PLOT_LAY, coloraxis_showscale=False)
        st.plotly_chart(fc, use_container_width=True)


# ============================================================
# PAGE 4 — CHURN RISK ANALYSIS
# ============================================================
elif page == PAGE_LABELS[3]:
    st.title("\u26a0\ufe0f Churn Risk Analysis")
    st.markdown("**XGBoost Classifier — Identify At-Risk Customers**")
    st.markdown("---")

    total_c    = len(churn_df)
    churned    = int(churn_df["Churned"].sum())
    active     = total_c - churned
    churn_rate = (churned / total_c) * 100 if total_c > 0 else 0
    high_risk  = int((churn_df["Risk_Tier"] == "Critical").sum()) if "Risk_Tier" in churn_df.columns else 0
    avg_prob   = churn_df["Churn_Probability"].mean() * 100

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("\U0001f4ca Churn Rate",    f"{churn_rate:.1f}%")
    c2.metric("\u26a0\ufe0f Churned",    f"{churned:,}")
    c3.metric("\u2705 Active",            f"{active:,}")
    c4.metric("\U0001f6a8 Critical Risk", f"{high_risk:,}")
    c5.metric("\U0001f4c9 Avg Churn Prob",f"{avg_prob:.1f}%")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("\U0001f967 Active vs Churned")
        cs = pd.DataFrame({"Status":["Active","Churned"],"Count":[active,churned]})
        fp = px.pie(cs, values="Count", names="Status",
                    color_discrete_sequence=["#27AE60","#E84E1B"], hole=0.45)
        fp.update_layout(**PLOT_LAY)
        st.plotly_chart(fp, use_container_width=True)

    with col2:
        st.subheader("\U0001f4ca Churn Probability Distribution")
        fh = px.histogram(churn_df, x="Churn_Probability", nbins=40,
                          color_discrete_sequence=["#E84E1B"],
                          labels={"Churn_Probability":"Probability","count":"Customers"})
        fh.update_layout(**PLOT_LAY)
        st.plotly_chart(fh, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        if "Risk_Tier" in churn_df.columns:
            st.subheader("\U0001f3af Risk Tier Breakdown")
            rc = churn_df["Risk_Tier"].value_counts().reset_index()
            rc.columns = ["Risk Tier","Count"]
            rc_map = {"Critical":"#E84E1B","High":"#F7941D","Medium":"#FBBA13","Low":"#27AE60"}
            fr = px.bar(rc, x="Risk Tier", y="Count", color="Risk Tier",
                        color_discrete_map=rc_map)
            fr.update_layout(**PLOT_LAY)
            st.plotly_chart(fr, use_container_width=True)

    with col4:
        if "Segment_Label" in churn_df.columns:
            st.subheader("\U0001f4ca Churn Rate by Segment")
            sc = churn_df.groupby("Segment_Label")["Churned"].mean().reset_index()
            sc.columns = ["Segment","Churn Rate"]
            sc["Churn Rate %"] = sc["Churn Rate"] * 100
            sc = sc.sort_values("Churn Rate %", ascending=False)
            fs = px.bar(sc, x="Segment", y="Churn Rate %",
                        color="Churn Rate %",
                        color_continuous_scale=["#27AE60","#FBBA13","#E84E1B"],
                        labels={"Churn Rate %":"Churn Rate (%)","Segment":""})
            fs.update_layout(**PLOT_LAY, coloraxis_showscale=False, xaxis_tickangle=-30)
            st.plotly_chart(fs, use_container_width=True)

    st.markdown("---")

    st.subheader("\U0001f6a8 High-Risk Customers")
    risk_opts = (["All"] + sorted(churn_df["Risk_Tier"].dropna().unique().tolist())
                 if "Risk_Tier" in churn_df.columns else ["All"])
    risk_filt = st.selectbox("Filter by Risk Tier:", risk_opts)

    dc2 = churn_df if risk_filt == "All" else churn_df[churn_df["Risk_Tier"] == risk_filt]
    dc2 = dc2.sort_values("Churn_Probability", ascending=False)
    dcc = [c for c in ["CustomerID","Recency","Frequency","Monetary",
                        "RFM_Score","Churn_Probability","Risk_Tier","Churned"]
           if c in dc2.columns]
    st.dataframe(dc2[dcc].head(200).reset_index(drop=True),
                 use_container_width=True, height=380)

    csv = dc2.to_csv(index=False).encode("utf-8")
    st.download_button("\U0001f4e5 Download Customer List",
                       data=csv, file_name="high_risk_customers.csv", mime="text/csv")

    # ── Live Churn Predictor ──────────────────────────────────
    st.markdown("---")
    st.subheader("\U0001f52e Live Churn Predictor")
    st.caption("Enter RFM values to get a real-time churn prediction")

    with st.form("churn_form"):
        f1,f2,f3 = st.columns(3)
        recency   = f1.number_input("Recency (days)", 0, 1000, 30)
        frequency = f2.number_input("Frequency (orders)", 1, 500, 5)
        monetary  = f3.number_input("Monetary (\u00a3)", 0.0, 200000.0, 1500.0)

        f4,f5,f6,f7 = st.columns(4)
        r_score   = f4.slider("R Score (1-5)", 1, 5, 3)
        f_score   = f5.slider("F Score (1-5)", 1, 5, 3)
        m_score   = f6.slider("M Score (1-5)", 1, 5, 3)
        rfm_score = f7.number_input("RFM Score", 1.0, 5.0, 3.0, step=0.01)

        predict_btn = st.form_submit_button("\U0001f680 Predict Churn", use_container_width=True)

    if predict_btn:
        if churn_model is None:
            st.warning("\u26a0\ufe0f Churn model not loaded. Check models/ folder.")
        else:
            try:
                # Build input row with correct column order
                row = {
                    "Recency"  : float(recency),
                    "Frequency": float(frequency),
                    "Monetary" : float(monetary),
                    "R_Score"  : float(r_score),
                    "F_Score"  : float(f_score),
                    "M_Score"  : float(m_score),
                    "RFM_Score": float(rfm_score),
                }
                if churn_feat_cols:
                    # Reorder / subset to the exact columns the model was trained on
                    vals = [row[c] for c in churn_feat_cols if c in row]
                    arr  = np.array(vals, dtype=np.float64).reshape(1, -1)
                else:
                    # Fallback: use the fixed order above
                    arr = np.array(list(row.values()), dtype=np.float64).reshape(1, -1)

                prob = float(churn_model.predict_proba(arr)[0][1])

                if prob >= 0.75:
                    risk = "\U0001f534 Critical"
                    reco = "Send immediate 20% discount voucher"
                    color = "#E84E1B"
                elif prob >= 0.50:
                    risk = "\U0001f7e0 High"
                    reco = "Send personalised email campaign"
                    color = "#F7941D"
                elif prob >= 0.25:
                    risk = "\U0001f7e1 Medium"
                    reco = "Include in loyalty rewards programme"
                    color = "#FBBA13"
                else:
                    risk = "\U0001f7e2 Low"
                    reco = "Customer healthy — standard engagement"
                    color = "#27AE60"

                r1,r2,r3 = st.columns(3)
                r1.metric("\U0001f3af Churn Probability", f"{prob*100:.1f}%")
                r2.metric("\u26a0\ufe0f Risk Level",     risk)
                r3.metric("\U0001f4a1 Recommended Action", reco)

                # Visual probability bar
                st.markdown(f"""
                <div style="margin-top:12px;">
                  <div style="color:#888;font-size:12px;margin-bottom:6px;">Churn Probability</div>
                  <div style="background:#1e1e2e;border-radius:8px;height:14px;overflow:hidden;">
                    <div style="width:{prob*100:.1f}%;background:{color};
                                height:100%;border-radius:8px;
                                transition:width 0.5s ease;"></div>
                  </div>
                  <div style="color:{color};font-size:13px;margin-top:5px;font-weight:600;">
                    {prob*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"\u274c Prediction error: {e}")


# ============================================================
# PAGE 5 — INVENTORY OPTIMIZATION
# ============================================================
elif page == PAGE_LABELS[4]:
    st.title("\U0001f4e6 Inventory Optimization")
    st.markdown("**EOQ \u00b7 Safety Stock \u00b7 ABC-XYZ Analysis \u00b7 Deadstock Risk**")
    st.markdown("---")

    # Correct KPIs: rows vs unique SKUs
    total_rows   = inventory.shape[0]            # 5296 — rows in file
    unique_skus  = inventory["StockCode"].nunique()  # 4614 — distinct products
    total_rev    = inventory["Total_Revenue"].sum()
    avg_eoq      = inventory["EOQ"].mean()
    at_risk_n    = (inventory["DeadStock_Risk"] != "Low").sum()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("\U0001f4cb Total Rows",     f"{total_rows:,}")
    c2.metric("\U0001f4e6 Unique SKUs",    f"{unique_skus:,}")
    c3.metric("\U0001f4b0 Total Revenue",  f"\u00a3{total_rev:,.0f}")
    c4.metric("\U0001f4d0 Avg EOQ",        f"{avg_eoq:,.0f} units")
    c5.metric("\u26a0\ufe0f At-Risk SKUs",f"{at_risk_n:,}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("\U0001f524 ABC Classification")
        abc_c = inventory["ABC_Class"].value_counts().reset_index()
        abc_c.columns = ["ABC Class","Count"]
        fp = px.pie(abc_c, values="Count", names="ABC Class", hole=0.4,
                    color="ABC Class",
                    color_discrete_map={"A":"#E84E1B","B":"#F7941D","C":"#FBBA13"})
        fp.update_layout(**PLOT_LAY)
        fp.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fp, use_container_width=True)

    with col2:
        st.subheader("\U0001f4ca XYZ Classification")
        xyz_c = inventory["XYZ_Class"].value_counts().reset_index()
        xyz_c.columns = ["XYZ Class","Count"]
        fx = px.bar(xyz_c, x="XYZ Class", y="Count",
                    color="Count", color_continuous_scale=HEAT)
        fx.update_layout(**PLOT_LAY, coloraxis_showscale=False)
        st.plotly_chart(fx, use_container_width=True)

    st.subheader("\U0001f321\ufe0f ABC-XYZ Revenue Matrix")
    piv_data = inventory.groupby(["ABC_Class","XYZ_Class"])["Total_Revenue"].sum().reset_index()
    pwide    = piv_data.pivot(index="ABC_Class", columns="XYZ_Class",
                              values="Total_Revenue").fillna(0)
    fm = px.imshow(pwide, color_continuous_scale=HEAT,
                   labels=dict(color="Revenue (\u00a3)"))
    fm.update_layout(**PLOT_LAY)
    st.plotly_chart(fm, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("\u26a0\ufe0f Deadstock Risk")
        ds_c = inventory["DeadStock_Risk"].value_counts().reset_index()
        ds_c.columns = ["Risk","Count"]
        ds_map = {"Low":"#27AE60","Medium":"#FBBA13","High":"#F7941D","Critical":"#E84E1B"}
        fd = px.bar(ds_c, x="Risk", y="Count", color="Risk",
                    color_discrete_map=ds_map)
        fd.update_layout(**PLOT_LAY)
        st.plotly_chart(fd, use_container_width=True)

    with col4:
        st.subheader("\U0001f3c6 Top 10 Products by Revenue")
        tp = inventory.nlargest(10, "Total_Revenue")
        ft = px.bar(tp, x="Total_Revenue", y="Description", orientation="h",
                    color="Total_Revenue", color_continuous_scale=HEAT,
                    labels={"Total_Revenue":"Revenue (\u00a3)","Description":""})
        ft.update_layout(**PLOT_LAY, coloraxis_showscale=False,
                         yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(ft, use_container_width=True)

    st.markdown("---")

    st.subheader("\U0001f4d0 EOQ vs Daily Demand")
    col5, col6 = st.columns(2)
    with col5:
        fe = px.scatter(
            inventory.head(800), x="Daily_Demand", y="EOQ",
            color="ABC_Class",
            color_discrete_map={"A":"#E84E1B","B":"#F7941D","C":"#FBBA13"},
            opacity=0.65, hover_data=["Description"],
            labels={"Daily_Demand":"Daily Demand (units)","EOQ":"EOQ (units)"}
        )
        fe.update_layout(**PLOT_LAY)
        st.plotly_chart(fe, use_container_width=True)

    with col6:
        fs2 = px.scatter(
            inventory.head(800), x="Safety_Stock", y="Reorder_Point",
            color="ABC_Class",
            color_discrete_map={"A":"#E84E1B","B":"#F7941D","C":"#FBBA13"},
            opacity=0.65, hover_data=["Description"],
            labels={"Safety_Stock":"Safety Stock","Reorder_Point":"Reorder Point"}
        )
        fs2.update_layout(**PLOT_LAY)
        st.plotly_chart(fs2, use_container_width=True)

    st.markdown("---")

    st.subheader("\U0001f50d Inventory Details Table")
    abc_f = st.multiselect("Filter by ABC Class:", ["A","B","C"], default=["A","B","C"])
    inv_d = inventory[inventory["ABC_Class"].isin(abc_f)]
    dcols = [c for c in ["StockCode","Description","ABC_Class","XYZ_Class","ABC_XYZ",
                          "Daily_Demand","EOQ","Safety_Stock","Reorder_Point",
                          "DeadStock_Risk","Total_Revenue"] if c in inv_d.columns]
    st.dataframe(
        inv_d[dcols].sort_values("Total_Revenue", ascending=False).reset_index(drop=True),
        use_container_width=True, height=420
    )


# ============================================================
# PAGE 6 — MODEL PERFORMANCE  (no MLflow)
# ============================================================
elif page == PAGE_LABELS[5]:
    st.title("\U0001f9ea Model Performance")
    st.markdown("**Trained Model Registry & File Status**")
    st.markdown("---")

    # Summary table
    st.subheader("\U0001f4ca Trained Models Summary")
    md = pd.DataFrame({
        "Model"         :["XGBoost Churn Classifier","LightGBM Forecasting Model",
                          "K-Means Segmentation"],
        "Task"          :["Churn Prediction","Demand Forecasting","Customer Segmentation"],
        "Primary Metric":["AUC-ROC","MAPE","Silhouette Score"],
        "Target"        :["\u2265 0.90","\u2264 8%","\u2265 0.50"],
        "Status"        :["\u2705 Trained","\u2705 Trained","\u2705 Trained"],
    })
    st.dataframe(md, use_container_width=True)

    st.markdown("---")

    # File status
    st.subheader("\U0001f4c1 Model File Status")
    model_files = {
        "churn_model.pkl"                  : "XGBoost Churn Model",
        "churn_features.pkl"               : "Churn Training Features",
        "lightgbm_forecasting_model.pkl"   : "LightGBM Forecast Model",
        "customer_segmentation_model.pkl"  : "K-Means Segmentation Model",
        "customer_segmentation_scaler.pkl" : "Segmentation Scaler",
    }
    for fname, label in model_files.items():
        fpath = os.path.join(MODELS_PATH, fname)
        if os.path.exists(fpath):
            kb = os.path.getsize(fpath) / 1024
            st.success(f"\u2705 **{label}** — `{fname}` ({kb:.1f} KB)")
        else:
            st.error(f"\u274c **{label}** — `{fname}` not found")

    st.markdown("---")

    # Live model details
    st.subheader("\U0001f52c Loaded Model Details")
    lm1, lm2 = st.columns(2)
    with lm1:
        if churn_model is not None:
            st.success("\u2705 Churn Model (XGBoost)")
            try:
                st.json({
                    "type"            : type(churn_model).__name__,
                    "n_estimators"    : int(getattr(churn_model,"n_estimators",0)),
                    "n_features"      : int(getattr(churn_model,"n_features_in_",0)),
                    "feature_columns" : churn_feat_cols,
                })
            except Exception: pass
        else:
            st.error("\u274c Churn model not loaded")

    with lm2:
        if forecast_model is not None:
            st.success("\u2705 Forecast Model (LightGBM)")
            try:
                st.json({
                    "type"        : type(forecast_model).__name__,
                    "n_estimators": int(getattr(forecast_model,"n_estimators",0)),
                    "n_features"  : int(getattr(forecast_model,"n_features_in_",0)),
                })
            except Exception: pass
        else:
            st.error("\u274c Forecast model not loaded")

    st.markdown("---")

    # Tech stack (no MLflow row)
    st.subheader("\U0001f6e0\ufe0f Technology Stack")
    ts = pd.DataFrame({
        "Layer"  : ["ML","ML","ML","Dashboard","Dashboard","API","Data","Data"],
        "Tool"   : ["XGBoost","LightGBM","Prophet","Streamlit","Plotly",
                    "FastAPI","Pandas","NumPy"],
        "Version": ["3.0.2","4.3","1.1.7","1.45.1","6.1.2","0.111+","2.3.0","2.2.6"],
        "Purpose": ["Churn Prediction","Demand Forecasting","Time-Series Baseline",
                    "Interactive Dashboard","Charts & Visualisations",
                    "REST Scoring API","Data Wrangling","Numerical Computing"],
    })
    st.dataframe(ts, use_container_width=True)

    st.markdown("---")

    # How to run
    st.subheader("\U0001f680 How to Run")
    tab1, tab2 = st.tabs(["\u25b6 Dashboard","\u26a1 FastAPI"])
    with tab1:
        st.code("streamlit run dashboard.py", language="bash")
        st.info("Opens at http://localhost:8501")
    with tab2:
        st.code("uvicorn api:app --reload --port 8000", language="bash")
        st.info("Swagger UI at http://localhost:8000/docs")
