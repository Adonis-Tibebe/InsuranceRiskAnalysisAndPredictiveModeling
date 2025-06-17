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
  - Results are used to evaluate the statistical significance of risk variation# Notebooks
  
  This directory contains Jupyter notebooks for exploratory data analysis (EDA), statistical evaluation, and predictive modeling of insurance risk.
  
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
  
  - **Severity_model.ipynb**  
    Builds and evaluates regression models to predict claim severity:
    - Data preprocessing and feature engineering
    - Model training and evaluation using RMSE and R² score
    - Visualization of results, including decision trees and SHAP summary
  
  - **claim_probability.ipynb**  
    Builds and evaluates classification models to predict the probability of a claim:
    - Data preprocessing and feature engineering
    - Model training and evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
    - Visualization of results, including ROC curves and confusion matrices
  
  - **PremiumOptimiztion.ipynb**  
    Calculates dynamic, risk-based premiums by combining predicted claim probability and severity:
    - Data loading and preprocessing
    - Generation of predictions using trained models
    - Computation of optimized premiums
    - Comparison to current premiums and visualization of results
  
  ## Usage
  
  Open each notebook in Jupyter or VS Code to interactively explore the analysis.  
  The notebooks expect data files in the [../Data/](cci:7://file:///c:/Users/adoni/Desktop/KAIM%20COURSE/Data:0:0-0:0) directory and use utility functions from the `../src/` directory.
  
  ## Requirements
  
  See the root `requirements.txt` for dependencies.
  - Provides evidence-based support for pricing and underwriting decisions

## Usage

Open each notebook in Jupyter or VS Code to interactively explore the analysis.  
The notebooks expect data files in the `../Data/` directory and use utility functions from the `../src/` directory.

## Requirements

See the root `requirements.txt` for dependencies.
