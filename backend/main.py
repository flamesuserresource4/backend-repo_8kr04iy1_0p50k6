from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import io
from datetime import datetime, timedelta

app = FastAPI(title="Partner Performance Reporting API")

# CORS for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUIRED_COLUMNS = [
    "Date",
    "C2C",
    "NonSpider C2C",
    "Spider C2C",
    "blank",
    "Emails",
    "Spam Emails",
    "Calls",
    "Answered -20Calls",
    "Answered +20Calls",
    "Not Answered Calls",
]


def format_currency(value: float) -> str:
    try:
        return f"£{value:,.2f}"
    except Exception:
        return "£0.00"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    start_date: Optional[str] = Form(default=None),
    end_date: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    # Read CSV into pandas
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    # Validate columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return {
            "ok": False,
            "error": f"Missing required columns: {', '.join(missing)}",
        }

    # Parse date (DD/MM/YYYY) and clean
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    # Ensure numeric columns
    numeric_cols = [
        "C2C",
        "NonSpider C2C",
        "Spider C2C",
        "Emails",
        "Spam Emails",
        "Calls",
        "Answered -20Calls",
        "Answered +20Calls",
        "Not Answered Calls",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Drop the 'blank' column if present
    if "blank" in df.columns:
        df = df.drop(columns=["blank"])

    # Revenue formula
    df["Revenue"] = (
        df["NonSpider C2C"] * 0.12
        + df["Answered +20Calls"] * 7.50
        + df["Emails"] * 5.00
    )

    # Sort by date
    df = df.sort_values("Date")

    # Determine available date range
    min_date = pd.to_datetime(df["Date"].min()).date()
    max_date = pd.to_datetime(df["Date"].max()).date()

    # Determine selected date range
    if start_date and end_date:
        try:
            sel_start = datetime.fromisoformat(start_date).date()
            sel_end = datetime.fromisoformat(end_date).date()
        except ValueError:
            sel_start = max_date - timedelta(days=6)
            sel_end = max_date
    else:
        # default last 7 days available
        sel_end = max_date
        sel_start = max(min_date, max_date - timedelta(days=6))

    mask = (df["Date"].dt.date >= sel_start) & (df["Date"].dt.date <= sel_end)
    period_df = df.loc[mask].copy()

    period_len = (sel_end - sel_start).days + 1
    prev_end = sel_start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=period_len - 1)
    prev_mask = (df["Date"].dt.date >= prev_start) & (df["Date"].dt.date <= prev_end)
    prev_df = df.loc[prev_mask].copy()

    def sum_or_zero(d: pd.DataFrame, col: str) -> float:
        return float(d[col].sum()) if not d.empty else 0.0

    # KPIs
    billable_clicks = sum_or_zero(period_df, "NonSpider C2C")
    billable_calls = sum_or_zero(period_df, "Answered +20Calls")
    total_emails = sum_or_zero(period_df, "Emails")
    total_revenue = float(period_df["Revenue"].sum()) if not period_df.empty else 0.0

    prev_billable_clicks = sum_or_zero(prev_df, "NonSpider C2C")
    prev_billable_calls = sum_or_zero(prev_df, "Answered +20Calls")
    prev_total_emails = sum_or_zero(prev_df, "Emails")
    prev_total_revenue = float(prev_df["Revenue"].sum()) if not prev_df.empty else 0.0

    def pct_change(curr: float, prev: float) -> float:
        if prev == 0:
            return 100.0 if curr > 0 else 0.0
        return (curr - prev) / prev * 100.0

    # Daily revenue series
    daily = (
        period_df.groupby(period_df["Date"].dt.date)["Revenue"].sum().reset_index()
    )
    daily.rename(columns={"Date": "date", "Revenue": "revenue"}, inplace=True)

    # Monthly aggregation from entire dataset
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("Month").agg(
        Total_Clicks=("NonSpider C2C", "sum"),
        Total_Calls=("Answered +20Calls", "sum"),
        Total_Emails=("Emails", "sum"),
        Total_Billing=("Revenue", "sum"),
    ).reset_index()
    monthly["MonthLabel"] = monthly["Month"].dt.strftime("%Y-%m")

    monthly_table = [
        {
            "Month": row["MonthLabel"],
            "Total Clicks": float(row["Total_Clicks"]),
            "Total Calls": float(row["Total_Calls"]),
            "Total Emails": float(row["Total_Emails"]),
            "Total Billing": float(row["Total_Billing"]),
        }
        for _, row in monthly.iterrows()
    ]

    response: Dict[str, Any] = {
        "ok": True,
        "available_date_range": {
            "min": min_date.isoformat(),
            "max": max_date.isoformat(),
        },
        "selected_date_range": {
            "start": sel_start.isoformat(),
            "end": sel_end.isoformat(),
        },
        "kpis": {
            "billable_clicks": {
                "total": billable_clicks,
                "delta_pct": pct_change(billable_clicks, prev_billable_clicks),
            },
            "billable_calls": {
                "total": billable_calls,
                "delta_pct": pct_change(billable_calls, prev_billable_calls),
            },
            "total_emails": {
                "total": total_emails,
                "delta_pct": pct_change(total_emails, prev_total_emails),
            },
            "total_revenue": {
                "total": total_revenue,
                "delta_pct": pct_change(total_revenue, prev_total_revenue),
                "formatted": format_currency(total_revenue),
            },
        },
        "daily_revenue": [
            {"date": d.strftime("%Y-%m-%d"), "revenue": float(v)} for d, v in zip(daily["date"], daily["revenue"])
        ],
        "monthly": {
            "labels": [m["MonthLabel"] for _, m in monthly.iterrows()],
            "revenue": [float(m["Total_Billing"]) for _, m in monthly.iterrows()],
            "table": monthly_table,
        },
    }

    return response
