"""Module for listing down additional custom functions required for production."""

import pandas as pd
import numpy as np

def binned_median_income(df):
    """Bin the selling price column using quantiles."""
    return pd.cut(
        df["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )