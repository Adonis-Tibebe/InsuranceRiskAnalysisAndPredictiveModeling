import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mticker


def drop_high_null_columns(data: pd.DataFrame, threshold: float = 0.5, verbose: bool = True) -> pd.DataFrame:
    """
    Drops columns from the DataFrame that have a proportion of null values greater than the specified threshold.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        threshold (float): Proportion of null values above which columns will be dropped (default is 0.5).
        verbose (bool): If True, prints the columns being dropped.

    Returns:
        pd.DataFrame: DataFrame with high-null columns removed.
    """
    null_ratio = data.isnull().mean()
    columns_with_high_nulls = null_ratio[null_ratio > threshold].index
    if verbose:
        print(f"Columns with more than {int(threshold*100)}% null values: {list(columns_with_high_nulls)}")
    return data.drop(columns=columns_with_high_nulls, axis=1)


def impute_missing_values(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Imputes missing values in the DataFrame:
      - For numeric columns (as detected by data.describe()), fills NaN with the mean value of the column.
      - For non-numeric/object columns, fills NaN with 'Unknown'.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        verbose (bool): If True, prints the columns being imputed.

    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    data = data.copy()
    numeric_cols = data.describe().columns
    non_numeric_cols = [col for col in data.columns if col not in numeric_cols]

    if verbose:
        print(f"Imputing numeric columns with mean: {list(numeric_cols)}")
        print(f"Imputing non-numeric columns with 'Unknown': {non_numeric_cols}")

    for col in numeric_cols:
        mean_value = data[col].mean()
        data[col] = data[col].fillna(mean_value)
    data[non_numeric_cols] = data[non_numeric_cols].fillna('Unknown')
    return data
    return data



def plot_box_and_dist(df, column):
    """
    Plots a boxplot without extreme outliers and a histogram with kernel density estimation (KDE) of the given column in the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing the column to plot.
        column (str): Name of the column to plot.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Apply log transformation for readability
    data = df[column].dropna()
    data_log = np.log1p(data)

    # Boxplot without extreme outliers (or use data_log for log-transformed boxplot)
    sns.boxplot(x=data_log, ax=axes[0], showfliers=True, color='skyblue')
    axes[0].set_title(f"Boxplot of {column} (Log Scale)")
    axes[0].set_xlabel(f"log({column} + 1)")

    # Distribution plot (histogram + KDE on log-transformed data)
    sns.histplot(data_log, ax=axes[1], kde=True, color='purple')
    axes[1].set_title(f"Distribution of {column} (Log Scale)")
    axes[1].set_xlabel(f"log({column} + 1)")

    plt.tight_layout()
    plt.show()

def plot_claim_frequency_by_count(df, date_col='TransactionMonth', claims_col='TotalClaims'):
    """
    Plots the number of claims made over time, with the x-axis representing the transaction month and the y-axis representing the number of claims.

    Parameters:
        df (pd.DataFrame): DataFrame containing the columns for transaction month and total claims.
        date_col (str): Name of the column containing the transaction month (default is 'TransactionMonth').
        claims_col (str): Name of the column containing the total claims (default is 'TotalClaims').

    Returns:
        None
    """
    # Step 1: Filter for entries where a claim was made (TotalClaims > 0)
    df_claims = df[df[claims_col] > 0]
    
    # Step 2: Group by TransactionMonth and count the number of claims
    claim_counts = df_claims.groupby(date_col).size().reset_index(name='ClaimCount')
    
    # Step 3: Plotting
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=claim_counts, x=date_col, y='ClaimCount', marker='o', color='darkorange')

    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(nbins=15))

    plt.title('Number of Claims Over Time', fontsize=14)
    plt.xlabel('Transaction Month')
    plt.ylabel('Number of Claims')
    plt.tight_layout()
    plt.show()
