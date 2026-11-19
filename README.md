# SmartSpend AI

SmartSpend AI is a local Python dashboard for exploring personal transactions. Start in **Synthetic demo** to explore fictional spending, then optionally upload a CSV. The app shows monthly gross spending, category totals, top merchant groups, unusual expenses, and a next-month forecast when history supports one.

## Quick start

Python 3.12 is the recommended starting point. From this project directory on macOS or Linux, run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open [localhost on port 8501](http://127.0.0.1:8501). Stop the app with Ctrl+C. On Windows PowerShell, use `py -3.12 -m venv .venv` to create the environment and `.\.venv\Scripts\Activate.ps1` to activate it. The remaining commands are the same.

Installation downloads open-source dependencies. The running app binds to the loopback interface, disables Streamlit telemetry, and needs no API keys, accounts, or database. Uploads and edits are handled in memory and the app writes no transaction logs or upload files. Changing datasets or ending the session clears manual corrections. Keep personal exports outside the project directory.

## CSV format

Use a UTF-8, comma-separated file with `date`, `description`, and `amount` columns. Header case and surrounding spaces are ignored. Extra columns are accepted and discarded. Use ISO dates (`YYYY-MM-DD`), nonempty descriptions, and plain numeric amounts without currency symbols or thousands separators. The upload limit is 10 MB and 100,000 transactions.

```csv
date,description,amount,synthetic
2025-01-01,Alder Court Rent,-1275.00,yes
2025-01-02,Fictional Payroll,3800.00,yes
2025-01-05,Lantern Cafe,-12.50,yes
```

- **Negative** means an expense. Reverse the signs before uploading if your export uses positive charges.
- **Positive** means a credit, such as a deposit or refund. Credits appear separately and do not reduce gross spending.
- **Zero** means no activity and is excluded from spending and credit totals.
- Use one currency per file. Amounts are displayed without a currency symbol. Remove any outgoing transfers or other debits you want excluded from spending. Duplicate rows count separately.

Download the [synthetic demo CSV](data/synthetic_demo.csv) from the sidebar or use it directly from the project. All bundled data is fictional and marked synthetic.

## How the analysis works

### Categorization

TF-IDF converts each description into numeric features. Common words get less weight than words distinctive to the training examples. The vectorizer uses both single words and adjacent word pairs. A multiclass logistic regression classifier learns how those features relate to categories in [synthetic_training.csv](data/synthetic_training.csv). The confidence column displays the model's highest predicted class probability. Descriptions with no recognized terms go to **Other** with no score. Credits and zero rows receive fixed labels.

The small fictional dataset is for learning and demonstration. Probabilities are uncalibrated for real bank records and no real-world accuracy has been measured. Choose an expense and save a category in **Transactions** to update the dashboard for that session. Corrections leave the original model output visible and do not retrain the classifier. Top merchant groups use matching descriptions after trimming surrounding spaces and ignoring case. Merchant aliases and transaction-specific suffixes can split groups.

The app trains the small synthetic classifier at startup and reuses it. To save the fitted pipeline as a local joblib artifact for inspection or reuse, run:

```sh
python -m src.categorization
```

This writes `models/expense_classifier.joblib`, which is ignored by Git. Only load joblib files you trust.

### Anomaly detection

For each category with at least five expenses, calculate the 25th percentile (Q1), the 75th percentile (Q3), and their difference (IQR). Flag expenses strictly above `Q3 + 1.5 × IQR`. The table shows the amount and the threshold for review. Legitimate one-time purchases can be flagged, and small groups are skipped. This is descriptive anomaly detection for large expenses.

### Forecasting and evaluation

Gross spending is summed by calendar month. A credit-only month with records has zero spending. A month without any rows is treated as unknown. The app excludes the first and last observed months as potentially partial and keeps the latest consecutive run of interior months. That run must have at least six months, with four or more for the initial fit and the two most recent for evaluation. The CSV cannot prove an interior month is complete.

Linear regression fits a straight line to elapsed month numbers and their spending totals. During the backtest, the model predicts each of the two held-out months using only earlier months. It refits after the first held-out month is known. The naive baseline repeats the previous month's actual spending. Each displayed MAE is the average absolute error over those two predictions in the file's currency, so smaller is better. The final model refits on all eligible months and projects to the month after the last observed transaction month. This bridges the excluded final month. The one-step backtest does not directly measure that longer projection. Predictions are floored at zero. Seasonality, major changes, and incomplete data can make the estimate unreliable.

## Project layout

| Path | Role |
| --- | --- |
| `app.py` | Streamlit interface, session corrections, and Plotly charts |
| `src/transactions.py` | CSV parsing and validation |
| `src/categorization.py` | TF-IDF, logistic regression, and optional joblib export |
| `src/anomalies.py` | Category-based thresholds |
| `src/forecasting.py` | Monthly totals, forecast, and chronological backtesting |
| `data/` | Labeled synthetic training and demo CSVs |
| `tests/` | Unit tests using fictional fixtures |
| `.streamlit/config.toml` | Dark theme, loopback binding, and telemetry settings |

Run the unit tests from the project directory with the virtual environment active:

```sh
python -m unittest discover -s tests -v
```
