# Insurance Risk Analysis and Predictive Modeling

This project provides tools and analysis for understanding insurance portfolio risk and supporting predictive modeling.

## Structure

- **notebooks/**  
  Contains Jupyter notebooks for exploratory data analysis (EDA), including data cleaning, risk profiling, and visualization.

- **src/**  
  Source code modules with reusable functions for data processing, cleaning, and plotting.

- **tests/**  
  Unit tests for validating the core data processing and analysis functions.

- **Data/**  
  Directory for raw and cleaned datasets used in analysis. **This directory is version-controlled using DVC. Please see the [DVC documentation](https://dvc.org/doc) for more information on how to work with the data files.**

## Key Features

- Data cleaning and preprocessing for insurance datasets
- Loss ratio analysis by province, vehicle type, and gender
- Distribution and outlier analysis of key financial variables
- Temporal and segment-based claim analysis
- Visualizations for risk assessment and business insights
- Modular code and unit tests for reliability

## Getting Started

1. Install dependencies from `requirements.txt`.
2. Place your data files in the `Data/` directory. **Note: The data files in this directory are tracked using DVC. To add new data files, please use the `dvc add` command.**
3. Explore the analysis in `notebooks/EDA_analysis.ipynb`.
4. Run unit tests from the `tests/` directory:

   ```sh
   python -m unittest discover tests

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies.
