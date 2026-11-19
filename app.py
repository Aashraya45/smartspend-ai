"""SmartSpend AI's local Streamlit dashboard."""

from hashlib import sha256
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.anomalies import MIN_CATEGORY_EXPENSES, detect_anomalies
from src.categorization import OTHER, categorize, train_classifier
from src.forecasting import (
    HOLDOUT_MONTHS,
    MIN_TRAIN_MONTHS,
    forecast_next_month,
    monthly_spending,
    recent_complete_months,
)
from src.transactions import read_transactions

DEMO = Path(__file__).resolve().parent / "data" / "synthetic_demo.csv"
TEAL = "#5EEAD4"
PALETTE = [TEAL, "#818CF8", "#60A5FA", "#FBBF24", "#F472B6", "#34D399", "#C084FC", "#FB923C", "#94A3B8"]

st.set_page_config(page_title="SmartSpend AI", page_icon="📊", layout="wide")
st.markdown("""
<style>
    .block-container {max-width: 82rem; padding-top: 3rem;}
    [data-testid="stMetric"] {background: #111D30; border: 1px solid #243650;
        border-radius: 1rem; padding: 1rem;}
    [data-testid="stMetricValue"] {font-size: clamp(1.3rem, 2.8vw, 2rem);}
    @media (max-width: 640px) {
        .block-container {padding-top: 2.5rem;}
        [data-testid="stHorizontalBlock"] {flex-wrap: wrap;}
        [data-testid="stColumn"] {flex: 1 1 100% !important; min-width: 0;}
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def synthetic_model():
    """Reuse the fitted synthetic model without caching any transaction data."""
    return train_classifier()


def money(amount):
    """Format single-currency amounts without assuming a currency."""
    return f"{amount:,.2f}"


def chart(figure):
    """Apply shared styling and fit the Plotly chart to its container."""
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=12, b=10),
        height=300,
        xaxis_title=None,
        yaxis_title=None,
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def transaction_columns():
    """Set display formats shared by the transaction and anomaly tables."""
    return {
        "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
        "description": "Description",
        "amount": st.column_config.NumberColumn("Amount", format="localized"),
        "spend": st.column_config.NumberColumn("Expense", format="localized"),
        "threshold": st.column_config.NumberColumn("Category threshold", format="localized"),
        "category": "Category",
        "confidence": st.column_config.NumberColumn("Model confidence", format="percent"),
        "predicted_category": "Model category",
    }


st.title("SmartSpend AI")
st.write("Explore spending, review unusual expenses, and estimate next month’s total.")
demo_bytes = DEMO.read_bytes()
with st.sidebar:
    st.header("Data source")
    source = st.radio("Choose transaction data", ["Synthetic demo", "Upload CSV"])
    st.download_button("Download synthetic example", demo_bytes, DEMO.name, "text/csv")
    with st.expander("CSV format and privacy"):
        st.write("Use UTF-8 with date, description, and amount columns. Dates use YYYY-MM-DD.")
        st.write("Negative amounts are expenses, positive amounts are credits, and zero is ignored. Use a single currency.")
        st.write("Uploads are processed in memory by the app running on your device.")

if source == "Synthetic demo":
    contents = demo_bytes
    st.caption("Synthetic demo · Fictional merchants and transactions")
else:
    file = st.file_uploader("Upload a transaction CSV", type=["csv"])
    if file is None:
        st.session_state.pop("dataset_id", None)
        st.session_state.pop("corrections", None)
        st.info("Upload a CSV or choose Synthetic demo in the sidebar.")
        st.stop()
    contents = file.getvalue()

# Corrections are keyed by row within the active file and stay in this session.
dataset_id = (source, sha256(contents).hexdigest())
if st.session_state.get("dataset_id") != dataset_id:
    st.session_state.dataset_id = dataset_id
    st.session_state.corrections = {}
try:
    transactions = read_transactions(contents)
except ValueError as error:
    st.error(str(error))
    st.stop()

model = synthetic_model()
transactions = categorize(transactions, model)
transactions["predicted_category"] = transactions["category"]
categories = sorted([*model.named_steps["classifier"].classes_, OTHER])

# Render the correction widget in its tab before building charts from the edited data.
overview, review, unusual, outlook = st.tabs(
    ["Overview", "Transactions", "Anomalies", "Forecast"]
)
with review:
    st.subheader("Review transactions")
    st.caption("The classifier learned from fictional descriptions. Confidence is a model score and may be unreliable on real records.")
    expense_rows = transactions[transactions["amount"] < 0]
    if not expense_rows.empty:
        def expense_label(row_id):
            row = expense_rows.loc[row_id]
            return f"#{row_id + 1} · {row['date']:%Y-%m-%d} · {row['description']} · {money(row['amount'])}"

        selected = st.selectbox("Expense to correct", expense_rows.index, format_func=expense_label)
        original = transactions.at[selected, "category"]
        current = st.session_state.corrections.get(selected, original)
        with st.form(f"category_{dataset_id[1]}_{selected}"):
            proposed = st.selectbox("Category", categories, index=categories.index(current))
            if st.form_submit_button("Save category"):
                if proposed == original:
                    st.session_state.corrections.pop(selected, None)
                else:
                    st.session_state.corrections[selected] = proposed
                st.success("Category saved for this session.")

    for row_id, corrected_category in st.session_state.corrections.items():
        transactions.at[row_id, "category"] = corrected_category
    display = transactions[["date", "description", "amount", "category", "predicted_category", "confidence"]]
    st.dataframe(display, hide_index=True, column_config=transaction_columns())

expenses = transactions[transactions["amount"] < 0].copy()
expenses["spend"] = -expenses["amount"]
totals = monthly_spending(transactions)
anomalies = detect_anomalies(expenses)
flagged = anomalies[anomalies["is_anomaly"]].sort_values("spend", ascending=False)

with overview:
    spending = expenses["spend"].sum()
    credits = transactions.loc[transactions["amount"] > 0, "amount"].sum()
    cards = st.columns(4)
    cards[0].metric("Gross spending", money(spending))
    cards[1].metric("Credits", money(credits))
    cards[2].metric("Expenses", f"{len(expenses):,}")
    cards[3].metric("Unusual expenses", f"{len(flagged):,}")

    st.subheader("Monthly spending")
    trend = totals.to_frame()
    trend["month"] = totals.index.astype(str)
    figure = px.line(trend, x="month", y="spend", markers=True, color_discrete_sequence=[TEAL],
                     labels={"month": "Month", "spend": "Spending"})
    figure.update_xaxes(type="date")
    figure.update_traces(connectgaps=False, fill="tozeroy")
    chart(figure)
    st.caption("Gross outflows by calendar month. Gaps indicate months without records.")

    if expenses.empty:
        st.info("Category and merchant breakdowns appear when the file contains expenses.")
    else:
        category_totals = expenses.groupby("category")["spend"].sum().sort_values(ascending=False)
        merchant_totals = (
            expenses.groupby(expenses["description"].str.casefold())["spend"].sum().nlargest(8)
        )
        merchant_names = expenses.groupby(expenses["description"].str.casefold())["description"].first()
        merchant_totals.index = merchant_names.reindex(merchant_totals.index)
        left, right = st.columns(2)
        with left:
            st.subheader("Spending by category")
            figure = px.pie(category_totals, names=category_totals.index, values="spend",
                            hole=0.62, color_discrete_sequence=PALETTE)
            figure.update_layout(legend=dict(orientation="h", y=-0.1))
            chart(figure)
        with right:
            st.subheader("Top merchants")
            top = merchant_totals.sort_values()
            chart(px.bar(top, x="spend", y=top.index, orientation="h",
                         color_discrete_sequence=[TEAL], labels={"spend": "Spending", "y": "Merchant"}))
            st.caption("Merchant groups use matching descriptions, ignoring capitalization.")
        st.subheader("Spending insights")
        st.text(f"{category_totals.index[0]} accounted for {category_totals.iloc[0] / spending:.1%} "
                 f"of gross spending. Your highest-spend merchant group was {merchant_totals.index[0]}.")
        if len(totals) >= 2 and pd.notna(totals.iloc[-2]):
            difference = totals.iloc[-1] - totals.iloc[-2]
            st.write(f"Spending changed by {difference:+,.2f} from {totals.index[-2]} to {totals.index[-1]}. "
                     "The latest month may be partial.")

with unusual:
    st.subheader("Unusually large expenses")
    st.caption(f"Anomaly detection compares each expense with its category’s 75th percentile plus "
               f"1.5 times its interquartile range. A category needs at least {MIN_CATEGORY_EXPENSES} expenses.")
    if flagged.empty:
        st.info("No expenses crossed an eligible category threshold.")
    else:
        st.dataframe(flagged[["date", "description", "category", "spend", "threshold"]],
                     hide_index=True, column_config=transaction_columns())
    skipped = sorted(anomalies.loc[anomalies["threshold"].isna(), "category"].unique())
    if skipped:
        st.caption("Too few expenses to assess " + ", ".join(skipped) + ".")

with outlook:
    st.subheader("Next-month spending forecast")
    forecast = forecast_next_month(totals)
    required = MIN_TRAIN_MONTHS + HOLDOUT_MONTHS
    st.caption("The forecast targets the month after the last transaction month. "
               "The first and last observed months are excluded because they may be partial.")
    if forecast is None:
        available = len(recent_complete_months(totals))
        st.info(f"Forecast unavailable. Found {available} consecutive interior months. "
                f"At least {required} are needed, including {HOLDOUT_MONTHS} for backtesting.")
    else:
        cards = st.columns(3)
        cards[0].metric(f"Estimate for {forecast.month.strftime('%b %Y')}", money(forecast.spending))
        cards[1].metric("Linear regression MAE", money(forecast.regression_mae))
        cards[2].metric("Naive baseline MAE", money(forecast.naive_mae))
        st.caption("MAE is mean absolute error in the file’s currency. Each backtest predicts one "
                   "month from earlier months. The naive baseline repeats the previous month’s total.")
        holdout = forecast.holdout.rename(columns={
            "actual": "Actual", "regression": "Linear regression", "naive": "Naive baseline"
        })
        holdout.index = holdout.index.astype(str)
        st.dataframe(holdout, column_config={
            name: st.column_config.NumberColumn(format="localized") for name in holdout
        })
        st.caption("The backtest measures one-month predictions. The final estimate projects across the excluded last month.")
