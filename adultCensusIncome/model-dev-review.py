"""
I saw you remove outliers using iqr and zscore in your notebook code. 
Instead of removing outliers, it’s safer to cap extreme values using 1%–99% quantiles.
Outliers usually represent real-world behavior.

"""

import numpy as np
import pandas as pd

def handle_outliers_quantile(
    X,
    num_cols,
    lower_q=0.01,
    upper_q=0.99,
    verbose=True
):
    """
    Caps extreme values instead of removing them.
    Preserves real-world signals while stabilizing distributions.
    """

    X = X.copy()

    if verbose:
        print(f"--- Quantile-based outlier capping ---\n")

    for col in num_cols:
        if col == "income_binary":
            continue

        lower = X[col].quantile(lower_q)
        upper = X[col].quantile(upper_q)

        before = ((X[col] < lower) | (X[col] > upper)).sum()

        X[col] = X[col].clip(lower=lower, upper=upper)

        after = ((X[col] < lower) | (X[col] > upper)).sum()

        if verbose:
            print(f"{col}")
            print(f"  capped below {lower_q:.2%}, above {upper_q:.2%}")
            print(f"  extreme values before: {before}")
            print(f"  extreme values after : {after}\n")

    return X
