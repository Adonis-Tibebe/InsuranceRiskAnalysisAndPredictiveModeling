# Source Code

This directory contains Python modules for data processing and analysis.

## Main Modules

- **EDA_analysis.py**  
  Provides reusable functions for:
  - Dropping columns with high null values
  - Imputing missing values
  - Plotting distributions and boxplots
  - Visualizing claim frequency over time

- **claim_probability.py**  
  Provides functions for building and evaluating classification models to predict the probability of a claim.

- **PremiumOptimization.py**  
  Provides functions for calculating dynamic, risk-based premiums by combining predicted claim probability and severity.

- **Severity_model.py**  
  Provides functions for building and evaluating regression models to predict claim severity.

## Usage

Import functions from these modules in your notebooks or scripts:

```python
from EDA_analysis import drop_high_null_columns, impute_missing_values, plot_box_and_dist, plot_claim_frequency_by_count
from claim_probability import train_claim_probability_model, evaluate_claim_probability_model
from PremiumOptimization import calculate_optimized_premiums
from Severity_model import train_severity_model, evaluate_severity_model