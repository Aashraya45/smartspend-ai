"""Summarize calendar months and evaluate a one-month-ahead spending forecast."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

MIN_TRAIN_MONTHS = 4
HOLDOUT_MONTHS = 2


@dataclass(frozen=True)
class Forecast:
    month: pd.Period
    spending: float
    regression_mae: float
    naive_mae: float
    holdout: pd.DataFrame


def monthly_spending(transactions: pd.DataFrame) -> pd.Series:
    """Aggregate gross outflows, keeping unobserved months as missing."""
    months = transactions["date"].dt.to_period("M")
    outflows = -transactions["amount"].clip(upper=0)
    totals = outflows.groupby(months).sum().rename("spend")
    if totals.empty:
        return totals
    calendar = pd.period_range(totals.index.min(), totals.index.max(), freq="M")
    return totals.reindex(calendar)


def recent_complete_months(totals: pd.Series) -> pd.Series:
    """Exclude statement edge months and use the latest stretch without gaps."""
    interior = totals.iloc[1:-1]
    gaps = np.flatnonzero(interior.isna())
    return interior.iloc[gaps[-1] + 1:] if len(gaps) else interior


def forecast_next_month(totals: pd.Series) -> Forecast | None:
    """Backtest on interior months, then project past the final observed month."""
    history = recent_complete_months(totals)
    if len(history) < MIN_TRAIN_MONTHS + HOLDOUT_MONTHS:
        return None

    # A single feature measures elapsed months. Each fit sees only prior months.
    time_steps = np.arange(len(history)).reshape(-1, 1)
    actual = history.to_numpy()
    holdout = history.iloc[-HOLDOUT_MONTHS:].to_frame("actual")
    regression_predictions = []
    naive_predictions = []
    for position in range(len(history) - HOLDOUT_MONTHS, len(history)):
        model = LinearRegression().fit(time_steps[:position], actual[:position])
        regression_predictions.append(max(0.0, float(model.predict([[position]])[0])))
        naive_predictions.append(actual[position - 1])
    holdout["regression"] = regression_predictions
    holdout["naive"] = naive_predictions

    target_month = totals.index[-1] + 1
    target_step = target_month.ordinal - history.index[0].ordinal
    model = LinearRegression().fit(time_steps, actual)
    return Forecast(
        month=target_month,
        spending=max(0.0, float(model.predict([[target_step]])[0])),
        regression_mae=float(mean_absolute_error(holdout["actual"], holdout["regression"])),
        naive_mae=float(mean_absolute_error(holdout["actual"], holdout["naive"])),
        holdout=holdout,
    )
