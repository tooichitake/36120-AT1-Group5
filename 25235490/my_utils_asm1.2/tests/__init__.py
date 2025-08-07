import pandas as pd
import numpy as np
import re

month_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

def missing_col(df):
    """Return a DataFrame of columns with missing values."""
    missing_col = df.isna().sum()
    missing_col_df = pd.DataFrame(missing_col[missing_col > 0])
    return missing_col_df

def get_numerical_and_categorical_columns(df):
    """Identify numeric and categorical column names."""
    numeric_cols = df.select_dtypes(exclude=['object']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return numeric_cols, categorical_cols

def parse_height(s):
    """Convert Excel‑mangled height strings into total inches."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    m = re.match(r"^(\d{1,2})[-/](\D+)$", s)
    if m:
        d, mth = m.groups()
        key = mth[:3].lower()
        if key in month_map:
            return month_map[key] * 12 + int(d)
    m2 = re.match(r"^(\D+)[-/](\d{1,2})$", s)
    if m2:
        mth, d = m2.groups()
        key = mth[:3].lower()
        if key in month_map:
            return month_map[key] * 12 + int(d)
    if s.isdigit():
        val = int(s)
        if 4 <= val <= 8:
            return val * 12
        if 50 <= val <= 110:
            return val
    return np.nan
