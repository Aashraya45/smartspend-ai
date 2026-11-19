"""Flag unusually large expenses using an explainable category threshold."""

import pandas as pd

MIN_CATEGORY_EXPENSES = 5


def detect_anomalies(expenses: pd.DataFrame) -> pd.DataFrame:
    """Flag purchase sizes above Q3 + 1.5 × IQR in categories with enough examples."""
    result = expenses.copy()
    category_sizes = result.groupby("category")["spend"]
    lower = category_sizes.transform("quantile", q=0.25)
    upper = category_sizes.transform("quantile", q=0.75)
    enough_data = category_sizes.transform("size") >= MIN_CATEGORY_EXPENSES
    result["threshold"] = (upper + 1.5 * (upper - lower)).where(enough_data)
    result["is_anomaly"] = result["spend"] > result["threshold"]
    return result
