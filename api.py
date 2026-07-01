# ============================================================
# NeuralRetail — FastAPI Scoring API
# Amdox Technologies | Data Science & Analytics
# Group 6 — Amdox Internship
# ============================================================

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import numpy as np
import pandas as pd
import os

# ============================================================
# Initialize FastAPI app
# ============================================================
app = FastAPI(
    title="NeuralRetail API",
    description="""
**AI-Powered Retail Intelligence API** — Amdox Technologies | Group 6 Internship

### Endpoints
| Category            | Endpoint                       | Method |
|---------------------|-------------------------------|--------|
| Health              | `/health`                     | GET    |
| Churn Prediction    | `/predict/churn`              | POST   |
| Demand Forecasting  | `/predict/demand`             | POST   |
| Customer Segments   | `/segments`                   | GET    |
| Inventory Analytics | `/inventory/summary`          | GET    |
| Top Products        | `/inventory/top-products`     | GET    |
| Customer Lookup     | `/customers/{customer_id}`    | GET    |
| Daily Sales         | `/sales/daily`                | GET    |
| Churn Batch         | `/predict/churn/batch`        | POST   |
""",
    version="2.0.0",
    contact={
        "name": "Amdox Data Science Team",
        "email": "ds@amdox.com",
        "url": "https://amdox.com"
    },
    license_info={"name": "MIT"}
)

# Allow CORS for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Load models and data at startup
# ============================================================
BASE        = os.path.dirname(os.path.abspath(__file__))
DATA        = os.path.join(BASE, "data")
MODELS_PATH = os.path.join(BASE, "models")

# Load churn model
churn_model    = joblib.load(os.path.join(MODELS_PATH, "churn_model.pkl"))
churn_features = joblib.load(os.path.join(MODELS_PATH, "churn_features.pkl"))

# Load forecast model
forecast_model = joblib.load(os.path.join(MODELS_PATH, "lightgbm_forecasting_model.pkl"))

# Load segmentation
seg_model  = joblib.load(os.path.join(MODELS_PATH, "customer_segmentation_model.pkl"))
seg_scaler = joblib.load(os.path.join(MODELS_PATH, "customer_segmentation_scaler.pkl"))

# Load dataframes
rfm       = pd.read_csv(os.path.join(DATA, "rfm_features.csv"))
churn_df  = pd.read_csv(os.path.join(DATA, "churn_predictions.csv"))
daily     = pd.read_csv(os.path.join(DATA, "daily_sales.csv"), parse_dates=["Date"])
inventory = pd.read_csv(os.path.join(DATA, "inventory_plan.csv"))

print("\u2705 NeuralRetail API — all models and data loaded successfully!")


# ============================================================
# Request / Response schemas
# ============================================================

class ChurnRequest(BaseModel):
    """Input data for single-customer churn prediction"""
    Recency   : float = Field(..., description="Days since last purchase", example=30)
    Frequency : float = Field(..., description="Number of purchases",      example=5)
    Monetary  : float = Field(..., description="Total amount spent (GBP)", example=1500.0)
    R_Score   : float = Field(..., description="Recency score (1-5)",      example=4)
    F_Score   : float = Field(..., description="Frequency score (1-5)",    example=3)
    M_Score   : float = Field(..., description="Monetary score (1-5)",     example=4)
    RFM_Score : float = Field(..., description="Combined RFM score",       example=3.67)

    class Config:
        json_schema_extra = {
            "example": {
                "Recency": 30, "Frequency": 5, "Monetary": 1500.0,
                "R_Score": 4,  "F_Score": 3,  "M_Score": 4, "RFM_Score": 3.67
            }
        }


class ChurnBatchRequest(BaseModel):
    """Batch input for churn prediction"""
    customers: List[ChurnRequest]


class DemandRequest(BaseModel):
    """Input data for demand forecasting"""
    Revenue_Lag_1  : float = Field(..., description="Revenue yesterday",          example=15000.0)
    Revenue_Lag_7  : float = Field(..., description="Revenue 7 days ago",         example=14000.0)
    Revenue_Lag_14 : float = Field(..., description="Revenue 14 days ago",        example=13000.0)
    Revenue_Lag_30 : float = Field(..., description="Revenue 30 days ago",        example=12000.0)
    Rolling_Mean_7 : float = Field(..., description="7-day rolling mean revenue", example=14500.0)
    Rolling_Mean_30: float = Field(..., description="30-day rolling mean",        example=13800.0)
    Rolling_Std_7  : float = Field(0.0,  description="7-day rolling std",         example=1200.0)
    DayOfWeek      : int   = Field(..., description="0=Mon, 6=Sun",               example=1)
    Month          : int   = Field(..., description="Month (1-12)",               example=11)
    Quarter        : int   = Field(..., description="Quarter (1-4)",              example=4)
    IsWeekend      : int   = Field(..., description="1=weekend, 0=weekday",       example=0)

    class Config:
        json_schema_extra = {
            "example": {
                "Revenue_Lag_1":15000.0, "Revenue_Lag_7":14000.0,
                "Revenue_Lag_14":13000.0,"Revenue_Lag_30":12000.0,
                "Rolling_Mean_7":14500.0,"Rolling_Mean_30":13800.0,
                "Rolling_Std_7":1200.0,  "DayOfWeek":1,
                "Month":11,              "Quarter":4,
                "IsWeekend":0
            }
        }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _get_churn_risk(prob: float) -> dict:
    """Map probability to risk tier and action."""
    if prob >= 0.75:
        return {
            "risk_tier"     : "Critical",
            "recommendation": "Send immediate retention offer — 20% discount voucher",
            "action_priority": "HIGH"
        }
    elif prob >= 0.50:
        return {
            "risk_tier"     : "High",
            "recommendation": "Send personalised email campaign with product recommendations",
            "action_priority": "MEDIUM"
        }
    elif prob >= 0.25:
        return {
            "risk_tier"     : "Medium",
            "recommendation": "Include in next loyalty rewards campaign",
            "action_priority": "LOW"
        }
    else:
        return {
            "risk_tier"     : "Low",
            "recommendation": "Customer is healthy — continue standard engagement",
            "action_priority": "NONE"
        }


# ============================================================
# ROOT & HEALTH ENDPOINTS
# ============================================================

@app.get("/", tags=["Info"])
def root():
    """Welcome message and API info"""
    return {
        "message"  : "Welcome to NeuralRetail API",
        "company"  : "Amdox Technologies",
        "team"     : "Group 6 — Amdox Internship 2026",
        "version"  : "2.0.0",
        "docs"     : "/docs",
        "redoc"    : "/redoc"
    }


@app.get("/health", tags=["Info"])
def health_check():
    """Health check — confirms API and models are running"""
    return {
        "status": "healthy",
        "models": {
            "churn_model"         : "loaded",
            "forecast_model"      : "loaded",
            "segmentation_model"  : "loaded",
        },
        "data": {
            "customers"     : int(rfm.shape[0]),
            "daily_records" : int(daily.shape[0]),
            "sku_count"     : int(inventory.shape[0]),
        }
    }


# ============================================================
# CHURN PREDICTION ENDPOINTS
# ============================================================

@app.post("/predict/churn", tags=["Churn Prediction"])
def predict_churn(request: ChurnRequest):
    """
    Predict whether a single customer will churn.

    Returns:
    - `churn_probability`: Probability of churn (0.0 – 1.0)
    - `churn_prediction`: Binary label (1 = churned, 0 = active)
    - `risk_tier`: Critical / High / Medium / Low
    - `recommendation`: Actionable retention strategy
    """
    try:
        input_data = pd.DataFrame([{
            "Recency"  : request.Recency,
            "Frequency": request.Frequency,
            "Monetary" : request.Monetary,
            "R_Score"  : request.R_Score,
            "F_Score"  : request.F_Score,
            "M_Score"  : request.M_Score,
            "RFM_Score": request.RFM_Score,
        }])

        if churn_features is not None:
            input_data = input_data[churn_features]

        churn_proba = float(churn_model.predict_proba(input_data)[0][1])
        churn_pred  = int(churn_proba >= 0.5)
        risk_info   = _get_churn_risk(churn_proba)

        return {
            "churn_probability": round(churn_proba, 4),
            "churn_prediction" : churn_pred,
            **risk_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/churn/batch", tags=["Churn Prediction"])
def predict_churn_batch(request: ChurnBatchRequest):
    """
    Batch churn prediction for multiple customers at once.

    Accepts a list of customer records and returns predictions for each.
    """
    try:
        results = []
        for customer in request.customers:
            inp = pd.DataFrame([{
                "Recency"  : customer.Recency,
                "Frequency": customer.Frequency,
                "Monetary" : customer.Monetary,
                "R_Score"  : customer.R_Score,
                "F_Score"  : customer.F_Score,
                "M_Score"  : customer.M_Score,
                "RFM_Score": customer.RFM_Score,
            }])
            if churn_features is not None:
                inp = inp[churn_features]
            prob = float(churn_model.predict_proba(inp)[0][1])
            pred = int(prob >= 0.5)
            results.append({
                "churn_probability": round(prob, 4),
                "churn_prediction" : pred,
                **_get_churn_risk(prob)
            })
        return {
            "count"      : len(results),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# DEMAND FORECASTING ENDPOINT
# ============================================================

@app.post("/predict/demand", tags=["Demand Forecasting"])
def predict_demand(request: DemandRequest):
    """
    Predict revenue demand for the next period using LightGBM.

    Uses lag features and rolling statistics. Returns:
    - `predicted_revenue`: Forecasted revenue (GBP)
    - `model`: Model name and version
    """
    try:
        input_data = pd.DataFrame([{
            "Revenue_Lag_1"  : request.Revenue_Lag_1,
            "Revenue_Lag_7"  : request.Revenue_Lag_7,
            "Revenue_Lag_14" : request.Revenue_Lag_14,
            "Revenue_Lag_30" : request.Revenue_Lag_30,
            "Rolling_Mean_7" : request.Rolling_Mean_7,
            "Rolling_Mean_30": request.Rolling_Mean_30,
            "Rolling_Std_7"  : request.Rolling_Std_7,
            "DayOfWeek"      : request.DayOfWeek,
            "Month"          : request.Month,
            "Quarter"        : request.Quarter,
            "IsWeekend"      : request.IsWeekend,
        }])

        predicted_revenue = float(forecast_model.predict(input_data)[0])

        # Simple confidence interval (±10% heuristic)
        return {
            "predicted_revenue"    : round(predicted_revenue, 2),
            "lower_bound_80pct"    : round(predicted_revenue * 0.90, 2),
            "upper_bound_80pct"    : round(predicted_revenue * 1.10, 2),
            "currency"             : "GBP",
            "model"                : "LightGBM — Lag + Rolling Features",
            "target_mape"          : "\u2264 8%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# CUSTOMER SEGMENT ENDPOINTS
# ============================================================

@app.get("/segments", tags=["Customer Intelligence"])
def get_segments():
    """
    Get customer segment summary.

    Returns count, revenue, average RFM score, and churn rate per segment.
    """
    try:
        seg_summary = rfm.groupby("Segment").agg(
            Customer_Count=("CustomerID", "count"),
            Total_Revenue =("Monetary",   "sum"),
            Avg_RFM_Score =("RFM_Score",  "mean"),
            Churn_Rate    =("Churned",    "mean")
        ).reset_index()

        seg_summary["Total_Revenue"] = seg_summary["Total_Revenue"].round(2)
        seg_summary["Avg_RFM_Score"] = seg_summary["Avg_RFM_Score"].round(3)
        seg_summary["Churn_Rate"]    = seg_summary["Churn_Rate"].round(4)

        return {
            "total_customers": int(len(rfm)),
            "segment_count"  : int(seg_summary.shape[0]),
            "segments"       : seg_summary.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/customers/{customer_id}", tags=["Customer Intelligence"])
def get_customer(customer_id: int):
    """
    Get detailed RFM and churn data for a specific customer by ID.
    """
    try:
        customer = rfm[rfm["CustomerID"] == customer_id]
        if customer.empty:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

        cust_dict = customer.iloc[0].to_dict()

        # Also get churn prediction if available
        churn_info = churn_df[churn_df["CustomerID"] == customer_id]
        if not churn_info.empty:
            ci = churn_info.iloc[0]
            cust_dict["Churn_Probability"] = round(float(ci.get("Churn_Probability", 0)), 4)
            cust_dict["Risk_Tier"]         = ci.get("Risk_Tier", "Unknown")

        return {"customer_id": customer_id, "data": cust_dict}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# INVENTORY ENDPOINTS
# ============================================================

@app.get("/inventory/summary", tags=["Inventory Optimization"])
def inventory_summary():
    """
    Get inventory summary statistics.

    Returns ABC-XYZ classification breakdown, EOQ averages,
    and deadstock risk counts.
    """
    try:
        abc_summary = inventory.groupby("ABC_Class").agg(
            SKU_Count   =("StockCode",   "count"),
            Total_Revenue=("Total_Revenue","sum"),
            Avg_EOQ     =("EOQ",          "mean"),
            Avg_Safety  =("Safety_Stock", "mean"),
        ).reset_index()
        abc_summary["Total_Revenue"] = abc_summary["Total_Revenue"].round(2)
        abc_summary["Avg_EOQ"]       = abc_summary["Avg_EOQ"].round(0)
        abc_summary["Avg_Safety"]    = abc_summary["Avg_Safety"].round(0)

        deadstock_summary = inventory["DeadStock_Risk"].value_counts().to_dict()

        return {
            "total_skus"       : int(inventory["StockCode"].nunique()),
            "total_revenue"    : round(float(inventory["Total_Revenue"].sum()), 2),
            "abc_breakdown"    : abc_summary.to_dict(orient="records"),
            "deadstock_risk"   : deadstock_summary,
            "avg_eoq"          : round(float(inventory["EOQ"].mean()), 0),
            "avg_safety_stock" : round(float(inventory["Safety_Stock"].mean()), 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inventory/top-products", tags=["Inventory Optimization"])
def top_products(
    n: int = Query(10, description="Number of top products to return", ge=1, le=100),
    abc_class: Optional[str] = Query(None, description="Filter by ABC class: A, B, or C")
):
    """
    Get top N products by revenue.

    Optionally filter by ABC classification (A = high value, B = medium, C = low).
    """
    try:
        inv_f = inventory.copy()
        if abc_class:
            inv_f = inv_f[inv_f["ABC_Class"] == abc_class.upper()]

        top = inv_f.nlargest(n, "Total_Revenue")[
            ["StockCode", "Description", "ABC_Class", "XYZ_Class",
             "Daily_Demand", "EOQ", "Safety_Stock", "Reorder_Point",
             "Total_Revenue", "DeadStock_Risk"]
        ]
        return {
            "count"   : len(top),
            "products": top.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# SALES ANALYTICS ENDPOINTS
# ============================================================

@app.get("/sales/daily", tags=["Sales Analytics"])
def get_daily_sales(
    limit : int = Query(30,  description="Number of records to return", ge=1, le=604),
    offset: int = Query(0,   description="Offset for pagination",       ge=0)
):
    """
    Get daily sales data with lag features and rolling averages.
    """
    try:
        data = daily.sort_values("Date", ascending=False).iloc[offset:offset+limit]
        data = data.fillna(0)
        data["Date"] = data["Date"].astype(str)
        return {
            "total_records": int(len(daily)),
            "returned"     : int(len(data)),
            "offset"       : offset,
            "records"      : data.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sales/summary", tags=["Sales Analytics"])
def get_sales_summary():
    """
    Get overall sales summary statistics.
    """
    try:
        daily_clean = daily.dropna(subset=["Revenue"])
        return {
            "total_revenue"      : round(float(daily_clean["Revenue"].sum()), 2),
            "total_orders"       : int(daily_clean["Orders"].sum()),
            "total_quantity"     : int(daily_clean["Quantity"].sum()),
            "avg_daily_revenue"  : round(float(daily_clean["Revenue"].mean()), 2),
            "max_daily_revenue"  : round(float(daily_clean["Revenue"].max()), 2),
            "min_daily_revenue"  : round(float(daily_clean["Revenue"].min()), 2),
            "date_from"          : str(daily_clean["Date"].min().date()),
            "date_to"            : str(daily_clean["Date"].max().date()),
            "total_days"         : int(len(daily_clean)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
