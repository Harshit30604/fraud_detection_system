import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing numeric values with median, drops rows if target is missing."""
    df = df.copy()
    if 'Class' in df.columns:
        df = df.dropna(subset=['Class'])
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df

def normalize_features(df: pd.DataFrame, cols=['Amount', 'Time']) -> tuple[pd.DataFrame, StandardScaler]:
    """Applies StandardScaler to specified columns."""
    df = df.copy()
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df, scaler

def drop_redundant(df: pd.DataFrame) -> pd.DataFrame:
    """Drops columns with >95% nulls or zero variance."""
    df = df.copy()
    # Drop columns with > 95% missing values
    threshold = len(df) * 0.05
    df = df.dropna(thresh=threshold, axis=1)
    
    # Drop zero variance columns
    variances = df.var(numeric_only=True)
    cols_to_drop = variances[variances == 0].index
    return df.drop(columns=cols_to_drop)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies domain-specific feature engineering for fraud detection."""
    df = df.copy()
    
    # 1. Log-transform Amount (handle right-skewness)
    # Using log1p to handle Amount == 0
    df['Amount_log'] = np.log1p(df['Amount'])
    
    # 2. Create hour_of_day from Time (Time is seconds from first transaction)
    df['hour_of_day'] = (df['Time'] // 3600) % 24
    
    # 3. Create amount_bins (ordinal feature: low, medium, high, very_high)
    df['amount_bins'] = pd.qcut(df['Amount'], q=4, labels=[1, 2, 3, 4])
    df['amount_bins'] = df['amount_bins'].astype(int)
    
    # 4. Rolling average of transaction amount per 10-row window
    # Assumes data is sorted by Time
    df = df.sort_values('Time').reset_index(drop=True)
    df['rolling_avg_amount_10'] = df['Amount'].rolling(window=10, min_periods=1).mean()
    
    return df
