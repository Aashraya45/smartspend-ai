"""Read and validate the app's transaction CSV format."""

import csv
from io import StringIO

import numpy as np
import pandas as pd

COLUMNS = ["date", "description", "amount"]
MAX_AMOUNT = 1_000_000_000_000
MAX_ROWS = 100_000


def read_transactions(contents: bytes) -> pd.DataFrame:
    """Return normalized transactions or raise a user-safe validation error."""
    try:
        reader = csv.reader(StringIO(contents.decode("utf-8-sig"), newline=""), strict=True)
        header = [name.strip().lower() for name in next(reader, [])]
        if len(set(header)) != len(header) or not set(COLUMNS).issubset(header):
            raise ValueError("Use unique headers including date, description, and amount.")
        rows = []
        for record_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"CSV record {record_number} has the wrong number of fields.")
            rows.append(row)
            if len(rows) > MAX_ROWS:
                raise ValueError(f"Use at most {MAX_ROWS:,} transactions.")
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError("Use a well-formed UTF-8 CSV file.") from error

    if not rows:
        raise ValueError("The CSV needs at least one transaction.")
    transactions = pd.DataFrame(rows, columns=header)[COLUMNS].copy()
    for column in COLUMNS:
        transactions[column] = transactions[column].str.strip()

    date_text = transactions["date"]
    dates = pd.to_datetime(date_text, format="%Y-%m-%d", errors="coerce")
    amounts = pd.to_numeric(transactions["amount"], errors="coerce")
    checks = [
        (dates.isna() | ~date_text.str.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}"),
         "date must be a valid YYYY-MM-DD date"),
        (transactions["description"].eq(""), "description must contain text"),
        (~np.isfinite(amounts) | amounts.abs().gt(MAX_AMOUNT),
         "amount must be a finite plain number within ±1 trillion"),
    ]
    for invalid, message in checks:
        if invalid.any():
            record = int(np.flatnonzero(invalid)[0]) + 1
            raise ValueError(f"Transaction {record}: {message}.")
    return transactions.assign(date=dates, amount=amounts.astype(float))
