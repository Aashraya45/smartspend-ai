"""Exercise the analytics with small synthetic examples."""

import unittest

import numpy as np
import pandas as pd

from src.anomalies import detect_anomalies
from src.categorization import categorize, train_classifier
from src.forecasting import forecast_next_month, monthly_spending, recent_complete_months
from src.transactions import read_transactions


class TransactionsTests(unittest.TestCase):
    def test_normalizes_headers_and_supports_quoted_descriptions(self):
        result = read_transactions(
            b'\xef\xbb\xbf Date , Description , Amount , synthetic\n2025-01-01,"Fictional Shop, East",-12.50,yes\n'
            b'2025-01-02,Fictional Payroll,100,yes\n2025-01-03,No activity,0,yes\n'
        )
        self.assertEqual(result.columns.tolist(), ["date", "description", "amount"])
        self.assertEqual(result["description"].iloc[0], "Fictional Shop, East")
        self.assertEqual(result["date"].iloc[0], pd.Timestamp("2025-01-01"))
        self.assertEqual(result["amount"].tolist(), [-12.5, 100.0, 0.0])

    def test_invalid_inputs(self):
        for contents in [
            b"",
            b"date,description,amount\n",
            b"date,description,cost\n2025-01-01,Shop,-10\n",
            b"date,description,amount,amount\n2025-01-01,Shop,-10,10\n",
            b"date,description,amount\n2025-02-30,Shop,-10\n",
            b"date,description,amount\n2025-2-01,Shop,-10\n",
            b"date,description,amount\n2025-01-01, ,10\n",
            b"date,description,amount\n2025-01-01,Shop,inf\n",
            b"date,description,amount\n2025-01-01,Shop,$9\n",
            b"date,description,amount\n2025-01-01,Shop,-10,extra\n",
            b'date,description,amount\n2025-01-01,"Shop,-10\n',
            b"\xff",
        ]:
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                read_transactions(contents)


class CategorizationTests(unittest.TestCase):
    def test_known_unseen_and_nonexpense_rows(self):
        transactions = pd.DataFrame({
            "description": ["Clover Grocery Market", "Lantern Cafe", "zzzz qqqq", "Credit", "Zero"],
            "amount": [-20, -9, -3, 100, 0],
        })
        result = categorize(transactions, train_classifier())
        self.assertEqual(result["category"].tolist(), ["Groceries", "Dining", "Other", "Credit", "No activity"])
        self.assertTrue(result["confidence"].iloc[:2].between(0, 1).all())
        self.assertTrue(result["confidence"].iloc[2:].isna().all())
        self.assertNotIn("category", transactions)


class AnomalyTests(unittest.TestCase):
    def test_threshold_is_category_specific_and_sparse_categories_are_skipped(self):
        expenses = pd.DataFrame({
            "category": ["Dining"] * 5 + ["Housing"] * 5 + ["Health"],
            "spend": [10, 10, 10, 10, 80] + [500] * 5 + [999],
        })
        result = detect_anomalies(expenses)
        self.assertEqual(result.index[result["is_anomaly"]].tolist(), [4])
        self.assertEqual(result["threshold"].iloc[4], 10)
        self.assertTrue(np.isnan(result["threshold"].iloc[-1]))

    def test_empty_expenses(self):
        expenses = pd.DataFrame({"category": pd.Series(dtype=str), "spend": pd.Series(dtype=float)})
        self.assertTrue(detect_anomalies(expenses).empty)


class ForecastTests(unittest.TestCase):
    def test_monthly_totals_preserve_unknown_months_and_credit_only_months(self):
        transactions = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-10", "2025-01-11", "2025-03-15"]),
            "amount": [-25, 100, 12],
        })
        totals = monthly_spending(transactions)
        self.assertEqual(totals.iloc[0], 25)
        self.assertTrue(pd.isna(totals.iloc[1]))
        self.assertEqual(totals.iloc[2], 0)

    def test_backtest_and_forecast_use_only_prior_eligible_months(self):
        totals = pd.Series(
            [9999, 100, 120, 140, 160, 180, 200, 9999],
            index=pd.period_range("2025-01", periods=8, freq="M"),
        )
        forecast = forecast_next_month(totals)
        self.assertEqual(forecast.month, pd.Period("2025-09", freq="M"))
        self.assertAlmostEqual(forecast.spending, 240)
        self.assertAlmostEqual(forecast.regression_mae, 0)
        self.assertAlmostEqual(forecast.naive_mae, 20)
        self.assertEqual(forecast.holdout["naive"].tolist(), [160, 180])

    def test_insufficient_or_gapped_history(self):
        totals = pd.Series(50.0, index=pd.period_range("2025-01", periods=9, freq="M"))
        self.assertIsNone(forecast_next_month(totals.iloc[:7]))
        totals.iloc[4] = np.nan
        self.assertEqual(len(recent_complete_months(totals)), 3)
        self.assertIsNone(forecast_next_month(totals))


if __name__ == "__main__":
    unittest.main()
