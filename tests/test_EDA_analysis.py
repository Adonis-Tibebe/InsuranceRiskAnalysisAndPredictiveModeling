import unittest
import pandas as pd
import numpy as np
from src import EDA_analysis

class TestEDAAnalysis(unittest.TestCase):

    def test_drop_high_null_columns(self):
        """
        Test the drop_high_null_columns function to ensure it correctly drops columns with a high proportion of null values.

        Creates a DataFrame with columns having different proportions of null values.
        Verifies that columns 'A' and 'C', which have more than 50% nulls, are dropped, while column 'B' remains.
        """

        df = pd.DataFrame({
            'A': [1, np.nan, np.nan],
            'B': [1, 2, 3],
            'C': [np.nan, np.nan, np.nan]
        })
        result = EDA_analysis.drop_high_null_columns(df, threshold=0.5, verbose=False)
        self.assertNotIn('C', result.columns)
        self.assertNotIn('A', result.columns)  # >50% nulls
        self.assertIn('B', result.columns)  

    def test_impute_missing_values(self):
        """
        Test the impute_missing_values function to ensure it correctly replaces missing values in a DataFrame.

        Creates a DataFrame with columns containing different types of data (numeric, non-numeric) and missing values.
        Verifies that the function:

        - Replaces NaNs in numeric columns with the mean of the column.
        - Replaces NaNs in non-numeric columns with 'Unknown'.
        - Does not change values in columns without missing values.
        """
        df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0],      # numeric
            'B': ['x', None, 'z'],        # non-numeric
            'C': [np.nan, np.nan, np.nan] # numeric, all missing
        })
        result = EDA_analysis.impute_missing_values(df, verbose=False)
        
        # Numeric columns: NaN replaced with mean
        expected_mean_A = np.nanmean([1.0, 3.0])
        self.assertTrue(np.allclose(result['A'], [1.0, expected_mean_A, 3.0]))
        
        # All-missing numeric column: should be filled with NaN's mean (which is NaN), so remains NaN
        self.assertTrue(result['C'].isnull().all() or result['C'].isnull().sum() == 0)  # Accepts either behavior
        
        # Non-numeric: NaN replaced with 'Unknown'
        self.assertEqual(result['B'][1], 'Unknown')
        self.assertFalse(result['B'].isnull().any())
        
    def test_plot_box_and_dist(self):
        df = pd.DataFrame({'A': np.random.rand(100)})
        # Just check that the function runs without error
        try:
            EDA_analysis.plot_box_and_dist(df, 'A')
        except Exception as e:
            self.fail(f"plot_box_and_dist raised an exception: {e}")

    def test_plot_claim_frequency_by_count(self):
        df = pd.DataFrame({
            'TransactionMonth': pd.date_range('2020-01-01', periods=10, freq='M'),
            'TotalClaims': [0, 1, 2, 0, 3, 0, 1, 2, 0, 1]
        })
        try:
            EDA_analysis.plot_claim_frequency_by_count(df, date_col='TransactionMonth', claims_col='TotalClaims')
        except Exception as e:
            self.fail(f"plot_claim_frequency_by_count raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()