# Notebooks

This directory contains Jupyter notebooks for exploratory data analysis (EDA) and statistical evaluation of insurance risk and predictive modeling strategies.

## Contents

- **EDA_analysis.ipynb**  
  Main EDA notebook covering:
  - Data cleaning and preprocessing
  - Loss ratio analysis by province, vehicle type, and gender
  - Distribution and outlier analysis of key financial variables
  - Temporal trends in claims
  - Segment analysis by vehicle make/model and geography
  - Visualizations and actionable insights

- **StatisticalAnalysis.ipynb**  
  Focuses on hypothesis testing to validate risk segmentation strategies:
  - KPI calculation for Claim Frequency, Claim Severity, and Margin
  - Hypothesis tests across Provinces, Postal Codes, and Gender
  - Statistical methods include Chi-squared and ANOVA tests
  - Results are used to evaluate the statistical significance of risk variation
  - Provides evidence-based support for pricing and underwriting decisions

## Usage

Open each notebook in Jupyter or VS Code to interactively explore the analysis.  
The notebooks expect data files in the `../Data/` directory and use utility functions from the `../src/` directory.

## Requirements

See the root `requirements.txt` for dependencies.
