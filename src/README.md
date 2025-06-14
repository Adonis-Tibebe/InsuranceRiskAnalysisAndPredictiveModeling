# Source Code

This directory contains Python modules for data processing and analysis.

## Main Modules

- **EDA_analysis.py**  
  Provides reusable functions for:
  - Dropping columns with high null values
  - Imputing missing values
  - Plotting distributions and boxplots
  - Visualizing claim frequency over time

## Usage

Import functions from `EDA_analysis.py` in your notebooks or scripts:

```python
from EDA_analysis import drop_high_null_columns, impute_missing_values, plot_box_and_dist, plot_claim_frequency_by_count
