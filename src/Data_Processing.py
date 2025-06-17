import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Define which categorical columns to encode and numeric transforms
BINARY_COLUMNS = ['IsVATRegistered', 'AlarmImmobiliser', 'TrackingDevice', 'NewVehicle', 'HadClaim']
ONEHOT_COLUMNS = ['Province', 'MainCrestaZone', 'VehicleType', 'CoverType', 'StatutoryRiskType', 'Gender']
LOG_COLUMNS = ['SumInsured', 'TotalPremium', 'CapitalOutstanding']
SCALE_COLUMNS = ['cubiccapacity', 'kilowatts', 'NumberOfDoors']


def load_and_clean_data(filepath):
    """
    Load CSV, drop duplicates, drop unwanted object columns, and impute missing values.
    Keeps only numeric columns, BINARY_COLUMNS, and ONEHOT_COLUMNS.
    """
    df = pd.read_csv(filepath)
    df = df.drop_duplicates(keep='first')

    # Identify object columns not needed (drop all except one-hot candidates)
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Always keep TransactionMonth for feature engineering
    keep_obj = ['TransactionMonth'] + [col for col in( ONEHOT_COLUMNS or BINARY_COLUMNS or LOG_COLUMNS or SCALE_COLUMNS) if col in df]
    drop_cols = [col for col in object_cols if col not in keep_obj]
    df = df.drop(columns=drop_cols)

    # Impute categorical NaNs with 'Unknown'
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Unknown')
    # Impute numeric NaNs with column mean
    for col in df.select_dtypes(include=['float64', 'int64', 'bool']).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    return df


def encode_categoricals(df):
    """
    Encode binary columns with LabelEncoder and one-hot encode specified columns.
    """
    df_encoded = df.copy()
    # Label encode binary flags
    for col in BINARY_COLUMNS:
        if col in df_encoded:
            lbl = LabelEncoder()
            df_encoded[col] = lbl.fit_transform(df_encoded[col].astype(str))
    # One-hot encode nominal categories
    df_encoded = pd.get_dummies(
        df_encoded,
        columns=[c for c in ONEHOT_COLUMNS if c in df_encoded.columns],
        prefix_sep='_', drop_first=True
    )
    return df_encoded


def scale_and_transform(df):
    """
    Apply log-transform to skewed financial columns and standard scale selected numeric columns.
    """
    df_scaled = df.copy()
    # Log-transform
    for col in LOG_COLUMNS:
        if col in df_scaled:
            df_scaled[col] = np.log1p(df_scaled[col])
    # Standard scale
    scaler = StandardScaler()
    cols = [c for c in SCALE_COLUMNS if c in df_scaled.columns]
    if cols:
        df_scaled[cols] = scaler.fit_transform(df_scaled[cols])
    return df_scaled

